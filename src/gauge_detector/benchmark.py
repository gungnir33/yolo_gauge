from __future__ import annotations

import csv
import statistics
from pathlib import Path

from .detector import GaugeDetector
from .io_utils import load_json
from .postprocess import box_iou


def match_boxes(predictions, ground_truth: list[list[float]], iou_threshold: float = 0.5) -> tuple[int, int, int]:
    candidates = sorted(
        ((box_iou(prediction.xyxy, truth), pi, gi) for pi, prediction in enumerate(predictions) for gi, truth in enumerate(ground_truth)),
        reverse=True,
    )
    used_predictions, used_truth = set(), set()
    tp = 0
    for iou, prediction_index, truth_index in candidates:
        if iou < iou_threshold:
            break
        if prediction_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            tp += 1
    return tp, len(predictions) - tp, len(ground_truth) - tp


def evaluate(
    detector: GaugeDetector,
    images_dir: str | Path,
    labels_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    labels = load_json(labels_path)
    total_tp = total_fp = total_fn = 0
    latencies = []
    for filename, truth in labels.items():
        result = detector.detect(str(Path(images_dir) / filename))
        tp, fp, fn = match_boxes(result.detections, truth, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        latencies.append(result.inference_ms)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordered = sorted(latencies)
    p95_index = max(0, int(round(0.95 * (len(ordered) - 1)))) if ordered else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": ordered[p95_index] if ordered else 0.0,
        "fps": 1000 / statistics.mean(latencies) if latencies and statistics.mean(latencies) else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def save_benchmark_csv(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "imgsz", "conf", "precision", "recall", "f1", "latency_ms", "fps", "tp", "fp", "fn"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def recommend(rows: list[dict], target_recall: float = 0.95) -> dict | None:
    if not rows:
        return None
    eligible = [row for row in rows if row["recall"] >= target_recall]
    pool = eligible or rows
    return max(pool, key=lambda row: (row["f1"], -row["latency_ms"]))
