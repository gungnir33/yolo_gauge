from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import Detection


def save_crops(image: np.ndarray, detections: list[Detection], output_dir: str | Path, stem: str) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    saved = []
    for index, item in enumerate(detections):
        x1, y1 = max(0, int(item.x1)), max(0, int(item.y1))
        x2, y2 = min(width, int(item.x2 + 0.999)), min(height, int(item.y2 + 0.999))
        crop = image[y1:y2, x1:x2]
        path = directory / f"{stem}_instrument_{index:02d}.jpg"
        if crop.size and cv2.imwrite(str(path), crop):
            saved.append(path)
    return saved
