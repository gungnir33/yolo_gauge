from __future__ import annotations

from .types import Detection


def box_iou(a: Detection | list[float] | tuple[float, ...], b: Detection | list[float] | tuple[float, ...]) -> float:
    aa = a.xyxy if isinstance(a, Detection) else a
    bb = b.xyxy if isinstance(b, Detection) else b
    ix1, iy1 = max(aa[0], bb[0]), max(aa[1], bb[1])
    ix2, iy2 = min(aa[2], bb[2]), min(aa[3], bb[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, aa[2] - aa[0]) * max(0.0, aa[3] - aa[1])
    area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicate_boxes(detections: list[Detection], threshold: float = 0.85) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(box_iou(candidate, existing) <= threshold for existing in kept):
            kept.append(candidate)
    return kept


def containment_ratio(a: Detection, b: Detection) -> float:
    """Return intersection divided by the smaller box area."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller_area = min(a.width * a.height, b.width * b.height)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def select_single_target(detections: list[Detection], config: dict) -> list[Detection]:
    """Keep the highest-confidence target, expanding to its largest nested box."""
    if not config.get("enabled", False) or not detections:
        return list(detections)
    threshold = float(config.get("containment_threshold", 0.90))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("containment_threshold must be between 0 and 1.")
    anchor = max(detections, key=lambda item: (item.confidence, item.width * item.height))
    nested = [item for item in detections if containment_ratio(anchor, item) >= threshold]
    selected = max(nested, key=lambda item: (item.width * item.height, item.confidence))
    return [selected]


def filter_geometry(
    detections: list[Detection], image_width: int, image_height: int, config: dict
) -> list[Detection]:
    if not config.get("enabled", False):
        return list(detections)
    image_area = image_width * image_height
    result = []
    for item in detections:
        area_ratio = item.width * item.height / image_area
        aspect_ratio = item.width / item.height
        if (
            config["min_area_ratio"] <= area_ratio <= config["max_area_ratio"]
            and config["min_aspect_ratio"] <= aspect_ratio <= config["max_aspect_ratio"]
        ):
            result.append(item)
    return result


def sort_detections(detections: list[Detection], x_tolerance: float = 2.0) -> list[Detection]:
    # Quantized x groups give a deterministic y tie-break for nearly aligned boxes.
    return sorted(detections, key=lambda item: (round(item.center[0] / x_tolerance), item.center[1], item.center[0]))
