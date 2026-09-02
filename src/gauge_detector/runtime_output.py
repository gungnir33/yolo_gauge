from __future__ import annotations

import numpy as np

from .preprocess import LetterboxTransform, restore_xyxy
from .types import Detection


def _make_detection(
    box: np.ndarray | list[float],
    score: float,
    transform: LetterboxTransform,
    image_shape: tuple[int, ...],
) -> Detection | None:
    x1, y1, x2, y2 = restore_xyxy(box, transform, image_shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(0, "instrument", float(score), x1, y1, x2, y2)


def decode_end2end_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    image_shape: tuple[int, ...],
    conf: float,
) -> list[Detection]:
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 6:
        raise ValueError(f"Expected end-to-end output shaped 1xNx6, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("End-to-end output must contain only finite values")
    detections = []
    for row in values[0]:
        if float(row[4]) <= float(conf):
            continue
        detection = _make_detection(row[:4], float(row[4]), transform, image_shape)
        if detection is not None:
            detections.append(detection)
    return detections


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def decode_raw_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    image_shape: tuple[int, ...],
    conf: float,
    iou: float,
    max_det: int,
) -> list[Detection]:
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 5:
        raise ValueError(f"Expected raw output shaped 1x(4+classes)xAnchors, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Raw output must contain only finite values")
    candidates = values[0].T
    scores = candidates[:, 4:].max(axis=1)
    selected = np.flatnonzero(scores > float(conf))
    if not len(selected):
        return []
    xywh = candidates[selected, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    candidate_scores = scores[selected]
    order = candidate_scores.argsort()[::-1]
    kept: list[int] = []
    while len(order) and len(kept) < int(max_det):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[_box_iou_one_to_many(boxes[current], boxes[remaining]) <= float(iou)]
    detections = []
    for index in kept:
        detection = _make_detection(boxes[index], float(candidate_scores[index]), transform, image_shape)
        if detection is not None:
            detections.append(detection)
    return detections
