import json

import numpy as np
import pytest

from gauge_detector.compare import compare_directory, compare_results
from gauge_detector.types import Detection, DetectionResult


def _result(path, detections, inference_ms=10.0):
    return DetectionResult(path, 100, 100, detections, inference_ms)


def test_compare_results_reports_hand_checked_iou():
    reference = _result("a.jpg", [Detection(0, "instrument", 0.9, 0, 0, 10, 10)])
    candidate = _result("a.jpg", [Detection(0, "instrument", 0.8, 5, 0, 15, 10)], 5)

    row = compare_results(reference, candidate, iou_threshold=0.8)

    assert row["iou"] == pytest.approx(1 / 3)
    assert row["reference_count"] == 1
    assert row["candidate_count"] == 1
    assert row["passed"] is False


def test_compare_results_marks_reference_detection_as_missed():
    reference = _result("missed.jpg", [Detection(0, "instrument", 0.9, 0, 0, 10, 10)])
    candidate = _result("missed.jpg", [])

    row = compare_results(reference, candidate)

    assert row["iou"] is None
    assert row["missed"] is True
    assert row["passed"] is False


def test_compare_directory_writes_json_and_markdown_summary(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.jpg").write_bytes(b"b")

    class FakeDetector:
        def __init__(self, config_path, **kwargs):
            self.reference = config_path == "reference.yaml"

        def detect_array(self, image):
            marker = int(image[0, 0, 0])
            if not self.reference and marker == 2:
                return _result("<ndarray>", [], 5)
            return _result(
                "<ndarray>",
                [Detection(0, "instrument", 0.9, 0, 0, 10, 10)],
                10 if self.reference else 5,
            )

    monkeypatch.setattr("gauge_detector.compare.GaugeDetector", FakeDetector)
    monkeypatch.setattr(
        "gauge_detector.compare.read_image",
        lambda path: np.full((2, 2, 3), 1 if path.name == "a.jpg" else 2, dtype=np.uint8),
    )

    report_path = compare_directory(
        "reference.yaml", "candidate.yaml", image_dir, tmp_path / "report", iou_threshold=0.8
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_images"] == 2
    assert report["summary"]["passed_images"] == 1
    assert report["summary"]["missed_images"] == ["b.jpg"]
    assert report["summary"]["candidate_mean_inference_ms"] == pytest.approx(5.0)
    assert report_path.with_suffix(".md").is_file()


def test_cli_parser_accepts_compare_backends_command():
    from gauge_detector.cli import build_parser

    args = build_parser().parse_args(
        [
            "compare-backends",
            "--reference",
            "configs/default.yaml",
            "--candidate",
            "configs/onnx.yaml",
            "--images",
            "images",
            "--output",
            "report",
        ]
    )

    assert args.command == "compare-backends"
    assert args.iou_threshold == 0.8
