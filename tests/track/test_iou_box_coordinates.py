"""Regression tests for persisted top-left xywh detection coordinates."""

from OTVision.domain.detection import Detection
from types import SimpleNamespace

from OTVision.track.tracker.tracker_plugin_iou import (
    ActiveIouTrack,
    BoundingBox,
    Coordinate,
    iou,
)


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


def test_long_gap_does_not_join_cars_moving_in_opposite_directions() -> None:
    # Real track 4363 from the 20630–20690 review clip: one car leaves left;
    # 37 missed frames later another car appears to its right.
    track = ActiveIouTrack(
        4363,
        SimpleNamespace(no=20637),
        detection(508.2, 123.4, 74.5, 26.7),
    )
    track.add_detection(SimpleNamespace(no=20638), detection(503.6, 119.8, 73.0, 26.5))
    track.add_detection(SimpleNamespace(no=20639), detection(496.2, 118.4, 75.5, 27.4))
    track.add_detection(SimpleNamespace(no=20640), detection(491.3, 118.8, 70.8, 24.1))

    other_car = detection(524.5, 118.4, 61.6, 25.2)
    assert not track.allows_direction_after_gap(20678, other_car)


def test_direction_gate_preserves_short_gap_and_forward_motion() -> None:
    track = ActiveIouTrack(1, SimpleNamespace(no=10), detection(100, 100, 60, 25))
    track.add_detection(SimpleNamespace(no=11), detection(95, 100, 60, 25))
    track.add_detection(SimpleNamespace(no=12), detection(90, 100, 60, 25))

    assert track.allows_direction_after_gap(15, detection(100, 100, 60, 25))
    assert track.allows_direction_after_gap(20, detection(70, 100, 60, 25))
    assert not track.allows_direction_after_gap(20, detection(115, 100, 60, 25))


def test_long_gap_without_reliable_direction_remains_eligible() -> None:
    track = ActiveIouTrack(1, SimpleNamespace(no=10), detection(100, 100, 60, 25))
    track.add_detection(SimpleNamespace(no=11), detection(100, 100, 60, 25))
    track.add_detection(SimpleNamespace(no=12), detection(100, 100, 60, 25))
    assert track.allows_direction_after_gap(50, detection(105, 100, 60, 25))


def test_long_gap_does_not_rejoin_a_later_box_of_the_other_car() -> None:
    # In the same clip the first competing box at 20687 is rejected, but
    # a box of that other car at 20688 slips past a direction-only threshold.
    track = ActiveIouTrack(
        4354, SimpleNamespace(no=20646), detection(459.914, 113.581, 70.330, 23.941)
    )
    track.add_detection(
        SimpleNamespace(no=20647), detection(457.980, 111.463, 66.560, 24.908)
    )
    track.add_detection(
        SimpleNamespace(no=20648), detection(455.098, 111.199, 65.232, 24.346)
    )

    assert not track.allows_direction_after_gap(
        20688, detection(472.605, 105.346, 63.633, 26.446)
    )
    assert not track.allows_direction_after_gap(
        20692, detection(457.664, 101.010, 57.498, 26.519)
    )
