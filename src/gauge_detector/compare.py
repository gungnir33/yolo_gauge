from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .detector import GaugeDetector
from .io_utils import read_image
from .postprocess import box_iou
from .types import DetectionResult


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compare_results(
    reference: DetectionResult,
    candidate: DetectionResult,
    iou_threshold: float = 0.8,
) -> dict[str, Any]:
    reference_count = len(reference.detections)
    candidate_count = len(candidate.detections)
    overlap: float | None = None
    if reference_count and candidate_count:
        overlap = box_iou(reference.detections[0], candidate.detections[0])
    counts_match = reference_count == candidate_count
    passed = counts_match and (reference_count == 0 or (overlap is not None and overlap >= iou_threshold))
    return {
        "image": Path(reference.image_path).name,
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "reference_boxes": [item.xyxy for item in reference.detections],
        "candidate_boxes": [item.xyxy for item in candidate.detections],
        "reference_confidences": [item.confidence for item in reference.detections],
        "candidate_confidences": [item.confidence for item in candidate.detections],
        "reference_inference_ms": reference.inference_ms,
        "candidate_inference_ms": candidate.inference_ms,
        "iou": overlap,
        "missed": reference_count > 0 and candidate_count == 0,
        "passed": passed,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Backend Comparison",
        "",
        f"- Images: {summary['total_images']}",
        f"- Passed: {summary['passed_images']}",
        f"- Missed: {', '.join(summary['missed_images']) or 'none'}",
        f"- Below IoU threshold: {', '.join(summary['below_iou_images']) or 'none'}",
        f"- Reference mean inference: {summary['reference_mean_inference_ms']:.2f} ms",
        f"- Candidate mean inference: {summary['candidate_mean_inference_ms']:.2f} ms",
        "",
        "| Image | Ref | Candidate | IoU | Ref ms | Candidate ms | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["images"]:
        overlap = "n/a" if row["iou"] is None else f"{row['iou']:.4f}"
        lines.append(
            f"| {row['image']} | {row['reference_count']} | {row['candidate_count']} | {overlap} | "
            f"{row['reference_inference_ms']:.2f} | {row['candidate_inference_ms']:.2f} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def compare_directory(
    reference_config: str | Path,
    candidate_config: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.8,
) -> Path:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported images found: {directory}")
    reference_detector = GaugeDetector(str(reference_config), warmup_runs=3)
    candidate_detector = GaugeDetector(str(candidate_config), warmup_runs=3)
    rows: list[dict[str, Any]] = []
    for path in paths:
        image = read_image(path)
        reference = reference_detector.detect_array(image)
        candidate = candidate_detector.detect_array(image)
        reference.image_path = candidate.image_path = str(path)
        rows.append(compare_results(reference, candidate, iou_threshold))
    report = {
        "reference_config": str(reference_config),
        "candidate_config": str(candidate_config),
        "iou_threshold": float(iou_threshold),
        "summary": {
            "total_images": len(rows),
            "passed_images": sum(bool(row["passed"]) for row in rows),
            "missed_images": [row["image"] for row in rows if row["missed"]],
            "count_mismatch_images": [
                row["image"] for row in rows if row["reference_count"] != row["candidate_count"]
            ],
            "below_iou_images": [
                row["image"] for row in rows if row["iou"] is not None and row["iou"] < iou_threshold
            ],
            "reference_mean_inference_ms": mean(row["reference_inference_ms"] for row in rows),
            "candidate_mean_inference_ms": mean(row["candidate_inference_ms"] for row in rows),
        },
        "images": rows,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json_path.with_suffix(".md").write_text(_markdown_report(report), encoding="utf-8")
    return json_path
