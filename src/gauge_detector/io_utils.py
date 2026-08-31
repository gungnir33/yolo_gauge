from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from .types import DetectionResult


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {file_path}")
    return data


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return file_path


def read_image(path: str | Path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {file_path}")
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {file_path}")
    return image


def result_to_dict(result: DetectionResult) -> dict[str, Any]:
    detections = []
    for index, item in enumerate(result.detections):
        detections.append(
            {
                "id": index,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "confidence": round(item.confidence, 4),
                "bbox_xyxy": [int(round(v)) for v in item.xyxy],
                "bbox_xywh": [int(round(v)) for v in item.xywh],
                "center": [int(round(v)) for v in item.center],
            }
        )
    return {
        "image": result.image_path,
        "width": result.image_width,
        "height": result.image_height,
        "num_detections": len(detections),
        "inference_ms": round(result.inference_ms, 2),
        "detections": detections,
    }

