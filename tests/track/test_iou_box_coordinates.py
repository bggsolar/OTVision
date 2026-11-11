"""Regression tests for persisted top-left xywh detection coordinates."""

from OTVision.domain.detection import Detection
from OTVision.track.tracker.tracker_plugin_iou import BoundingBox, Coordinate, iou


def detection(x: float, y: float, w: float, h: float) -> Detection:
    return Detection(label="car", conf=0.8, x=x, y=y, w=w, h=h)


def test_bbox_and_center_use_top_left_xywh_convention() -> None:
    observed = detection(100, 200, 20, 10)

    assert BoundingBox.from_xywh(observed).as_tuple() == (100, 200, 120, 210)
    assert Coordinate.center_of(observed) == Coordinate(110, 205)
    # Existing .otdet/.ottrk x and y values are not migrated or rewritten.
    assert observed.to_otdet()["x"] == 100
    assert observed.to_otdet()["y"] == 200


def test_changed_box_width_does_not_break_a_valid_match() -> None:
    previous = BoundingBox.from_xywh(detection(100, 100, 20, 20))
    narrower = BoundingBox.from_xywh(detection(110, 100, 10, 20))

    # The legacy center-based reconstruction yields 0.2 and misses this match.
    assert iou(previous, narrower) == 0.5
    assert iou(previous, narrower) >= 0.38


def test_changed_box_width_does_not_create_a_false_match() -> None:
    previous = BoundingBox.from_xywh(detection(100, 100, 20, 20))
    wider = BoundingBox.from_xywh(detection(110, 100, 30, 20))

    # The legacy center-based reconstruction yields about 0.429 instead.
    assert iou(previous, wider) == 0.25
    assert iou(previous, wider) < 0.38
