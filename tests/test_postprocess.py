import pytest

from gauge_detector.postprocess import (
    box_iou,
    containment_ratio,
    filter_geometry,
    remove_duplicate_boxes,
    select_single_target,
    sort_detections,
)
from gauge_detector.types import Detection
from gauge_detector.benchmark import recommend


def det(box, confidence=0.5):
    return Detection(0, "instrument", confidence, *box)


def test_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_duplicate_keeps_higher_confidence():
    boxes = [det([1, 1, 20, 20], 0.4), det([1, 1, 20, 20], 0.9), det([30, 1, 50, 20], 0.8)]
    kept = remove_duplicate_boxes(boxes)
    assert len(kept) == 2
    assert kept[0].confidence == 0.9


def test_containment_ratio_detects_nested_boxes_despite_low_iou():
    large = det([0, 0, 100, 100], 0.6)
    small = det([20, 20, 60, 60], 0.9)
    assert box_iou(large, small) == pytest.approx(0.16)
    assert containment_ratio(large, small) == 1.0


def test_single_target_keeps_highest_confidence_when_boxes_are_separate():
    boxes = [det([0, 0, 20, 20], 0.7), det([40, 40, 80, 80], 0.9)]
    assert select_single_target(boxes, {"enabled": True, "containment_threshold": 0.9}) == [boxes[1]]


def test_single_target_uses_largest_box_nested_with_confidence_anchor():
    large = det([0, 0, 100, 100], 0.6)
    anchor = det([20, 20, 60, 60], 0.9)
    unrelated = det([120, 0, 200, 80], 0.8)
    assert select_single_target(
        [anchor, unrelated, large], {"enabled": True, "containment_threshold": 0.9}
    ) == [large]


def test_single_target_does_not_treat_partial_overlap_as_nesting():
    anchor = det([0, 0, 50, 50], 0.9)
    partial = det([25, 0, 75, 50], 0.8)
    assert containment_ratio(anchor, partial) == 0.5
    assert select_single_target(
        [anchor, partial], {"enabled": True, "containment_threshold": 0.9}
    ) == [anchor]


def test_single_target_can_be_disabled_and_handles_empty_input():
    boxes = [det([0, 0, 20, 20], 0.7), det([40, 40, 80, 80], 0.9)]
    assert select_single_target([], {"enabled": True}) == []
    assert select_single_target(boxes, {"enabled": False}) == boxes


def test_sort_left_to_right_then_top_to_bottom():
    boxes = [det([50, 50, 60, 60]), det([10, 30, 20, 40]), det([10, 10, 20, 20])]
    ordered = sort_detections(boxes)
    assert [item.center for item in ordered] == [(15, 15), (15, 35), (55, 55)]


def test_geometry_filter_can_be_disabled_or_enabled():
    boxes = [det([0, 0, 10, 10]), det([0, 0, 90, 10])]
    disabled = {"enabled": False}
    enabled = {
        "enabled": True,
        "min_area_ratio": 0.001,
        "max_area_ratio": 0.5,
        "min_aspect_ratio": 0.5,
        "max_aspect_ratio": 2.0,
    }
    assert len(filter_geometry(boxes, 100, 100, disabled)) == 2
    assert filter_geometry(boxes, 100, 100, enabled) == [boxes[0]]


def test_benchmark_recommendation_prioritizes_f1_then_latency():
    rows = [
        {"recall": 0.96, "f1": 0.9, "latency_ms": 20},
        {"recall": 0.97, "f1": 0.9, "latency_ms": 10},
        {"recall": 0.90, "f1": 0.99, "latency_ms": 5},
    ]
    assert recommend(rows, 0.95) is rows[1]
