from dataclasses import dataclass

from OTVision.application.config import TrackConfig
from OTVision.application.get_current_config import GetCurrentConfig
from OTVision.domain.detection import Detection, TrackedDetection, TrackId
from OTVision.domain.frame import DetectedFrame, FrameNo, TrackedFrame
from OTVision.track.model.tracking_interfaces import IdGenerator, Tracker


@dataclass(frozen=True)
class IouParameters:
    sigma_l: float
    sigma_h: float
    sigma_iou: float
    t_min: int
    t_miss_max: int


@dataclass(frozen=True, repr=True)
class Coordinate:
    x: float
    y: float

    @staticmethod
    def center_of(detection: Detection) -> "Coordinate":
        return Coordinate(detection.x + detection.w / 2, detection.y + detection.h / 2)


@dataclass(frozen=True, repr=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @staticmethod
    def from_xywh(detection: Detection) -> "BoundingBox":
        """Calculates xyxy coordinates from Detection with xywh data:
            pixel values for the top-left corner x, y, width and height.

        Args:
            detection (Detection): detection to compute BoundingBox for.

        Returns:
            BoundingBox with pixel coordinates: xmin, ymin, xmax, ymax
        """
        return BoundingBox(
            detection.x,
            detection.y,
            detection.x + detection.w,
            detection.y + detection.h,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class ActiveIouTrack:
    # TODO check invariant -> at least one element in lists
    id: int
    frame_no: list[FrameNo]
    bboxes: list[BoundingBox]
    center: list[Coordinate]
    conf: list[float]
    classes: list[str]
    max_class: str
    max_conf: float
    first_frame: FrameNo
    last_frame: FrameNo
    track_age: int

    def __init__(
        self, id: TrackId, frame: "DetectedFrame", detection: "Detection"
    ) -> None:
        self.id = id
        self.frame_no = [frame.no]
        self.bboxes = [BoundingBox.from_xywh(detection)]
        self.center = [Coordinate.center_of(detection)]
        self.conf = [detection.conf]
        self.classes = [detection.label]
        self.max_class = detection.label
        self.max_conf = detection.conf
        self.first_frame = frame.no
        self.last_frame = frame.no
        self.track_age = 0

    def add_detection(self, frame: DetectedFrame, detection: Detection) -> None:
        self.frame_no.append(frame.no)
        self.bboxes.append(BoundingBox.from_xywh(detection))
        self.center.append(Coordinate.center_of(detection))
        self.conf.append(detection.conf)
        self.classes.append(detection.label)
        self.max_conf = max(self.max_conf, detection.conf)
        self.last_frame = max(self.last_frame, frame.no)
        self.track_age = 0

    @property
    def frame_span(self) -> int:
        return self.last_frame - self.first_frame

    def allows_direction_after_gap(self, frame_no: FrameNo, detection: Detection) -> bool:
        """Reject a long-gap match that reverses a recently observed motion.

        IoU alone compares the last observed boxes, even when dozens of frames
        have elapsed. This check applies only after more than five missed frames
        An isolated observation cannot establish identity across such a gap.
        With three recent observations, motion is measured per elapsed frame,
        even when individual detection frames were missed. This does not
        resolve ambiguous stationary or occluded objects.
        """
        if frame_no - self.last_frame - 1 <= 5:
            return True
        if len(self.center) == 1:
            return False
        if len(self.center) < 3 or self.frame_no[-1] - self.frame_no[-3] > 8:
            return True

        first, _, last = self.center[-3:]
        observation_span = self.frame_no[-1] - self.frame_no[-3]
        velocity_x = (last.x - first.x) / observation_span
        velocity_y = (last.y - first.y) / observation_span
        box = self.bboxes[-1]
        diagonal = ((box.xmax - box.xmin) ** 2 + (box.ymax - box.ymin) ** 2) ** 0.5
        speed = (velocity_x**2 + velocity_y**2) ** 0.5
        if speed < 0.03 * diagonal:
            return True

        candidate = Coordinate.center_of(detection)
        displacement_x = candidate.x - last.x
        displacement_y = candidate.y - last.y
        elapsed = frame_no - self.last_frame
        # An opposite-direction vehicle may enter at nearly the same position
        # a few frames after the first candidate was rejected. Its location
        # must also remain near the position predicted by recent motion.
        prediction_error = (
            (displacement_x - velocity_x * elapsed) ** 2
            + (displacement_y - velocity_y * elapsed) ** 2
        ) ** 0.5
        if prediction_error > 1.5 * diagonal:
            return False
        # Reject when the candidate lies more than a quarter box diagonal
        # behind the observed direction of travel.
        return velocity_x * displacement_x + velocity_y * displacement_y >= (
            -0.25 * diagonal * speed
        )

    def iou_with(self, detection: Detection) -> float:
        return iou(self.bboxes[-1], BoundingBox.from_xywh(detection))


def iou(
    bbox1: BoundingBox,
    bbox2: BoundingBox,
) -> float:
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (BoundingBox): first bounding box.
        bbox2 (BoundingBox): second bounding box.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    (xmin_1, ymin_1, xmax_1, ymax_1) = bbox1.as_tuple()
    (xmin_2, ymin_2, xmax_2, ymax_2) = bbox2.as_tuple()

    # get the overlap rectangle
    overlap_xmin = max(xmin_1, xmin_2)
    overlap_ymin = max(ymin_1, ymin_2)
    overlap_xmax = min(xmax_1, xmax_2)
    overlap_ymax = min(ymax_1, ymax_2)

    # check if there is an overlap
    if overlap_xmax - overlap_xmin <= 0 or overlap_ymax - overlap_ymin <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
    size_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

    size_intersection = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


class IouTracker(Tracker):
    @property
    def config(self) -> TrackConfig:
        return self._get_current_config.get().track

    def __init__(self, get_current_config: GetCurrentConfig):
        super().__init__()
        self._get_current_config = get_current_config
        self.active_tracks: list[ActiveIouTrack] = []

    @property
    def sigma_l(self) -> float:
        return self.config.iou.sigma_l

    @property
    def sigma_h(self) -> float:
        return self.config.iou.sigma_h

    @property
    def sigma_iou(self) -> float:
        return self.config.iou.sigma_iou

    @property
    def t_min(self) -> int:
        return self.config.iou.t_min

    @property
    def t_miss_max(self) -> int:
        return self.config.iou.t_miss_max

    def track_frame(
        self, frame: DetectedFrame, id_generator: IdGenerator
    ) -> TrackedFrame:

        detections = [d for d in frame.detections if d.conf >= self.sigma_l]
        tracked_detections: list[TrackedDetection] = []

        finished_track_ids: list[TrackId] = []
        discarded_track_ids: list[TrackId] = []

        saved_tracks: list[ActiveIouTrack] = []
        updated_tracks: list[ActiveIouTrack] = []
        new_tracks: list[ActiveIouTrack] = []

        # for each active track, check if detection with best iou match can be added
        for track in self.active_tracks:
            if detections:
                # get det with highest iou
                iou_pairs = (
                    (i, det, track.iou_with(det))
                    for i, det in enumerate(detections)
                    if track.allows_direction_after_gap(frame.no, det)
                )
                best = max(iou_pairs, key=lambda p: p[2], default=None)

                if best is not None and best[2] >= self.sigma_iou:
                    best_index, best_match, _ = best
                    track.add_detection(frame, best_match)
                    updated_tracks.append(track)

                    del detections[best_index]
                    tracked_detections.append(best_match.of_track(track.id, False))

            # if track was not updated, increase age (time without detection)
            # if max age is reached, check confidence, if sufficient finish that track
            if not updated_tracks or track is not updated_tracks[-1]:
                if track.track_age < self.t_miss_max:
                    track.track_age += 1
                    saved_tracks.append(track)
                elif track.max_conf >= self.sigma_h and track.frame_span >= self.t_min:
                    finished_track_ids.append(track.id)
                else:
                    discarded_track_ids.append(track.id)

        # start new track for each detection that could not be assigned
        for det in detections:
            track_id: TrackId = next(id_generator)
            new_tracks.append(ActiveIouTrack(track_id, frame, det))
            tracked_detections.append(det.of_track(track_id, True))

        self.active_tracks = updated_tracks + saved_tracks + new_tracks

        return TrackedFrame(
            no=frame.no,
            occurrence=frame.occurrence,
            source=frame.source,
            output=frame.output,
            detections=tracked_detections,
            image=frame.image,
            finished_tracks=set(finished_track_ids),
            discarded_tracks=set(discarded_track_ids),
        )
