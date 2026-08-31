from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


# OpenCV BGR palette. Adjacent detections deliberately use strongly
# contrasting colors and the palette cycles for images with many instruments.
BOX_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
    (255, 128, 0),  # light blue
)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    colors: tuple[tuple[int, int, int], ...] = BOX_COLORS,
    thickness: int = 3,
    draw_numbers: bool = True,
) -> np.ndarray:
    if not colors:
        raise ValueError("At least one bounding-box color is required.")
    output = image.copy()
    height, width = output.shape[:2]
    for index, item in enumerate(detections):
        color = colors[index % len(colors)]
        x1 = min(max(int(round(item.x1)), 0), width - 1)
        y1 = min(max(int(round(item.y1)), 0), height - 1)
        x2 = min(max(int(round(item.x2)), 0), width - 1)
        y2 = min(max(int(round(item.y2)), 0), height - 1)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        if draw_numbers:
            label = str(index + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            label_top = max(0, y1 - text_height - baseline - 8)
            label_bottom = label_top + text_height + baseline + 8
            label_right = min(width - 1, x1 + text_width + 12)
            cv2.rectangle(output, (x1, label_top), (label_right, label_bottom), color, -1)
            cv2.putText(
                output,
                label,
                (x1 + 6, label_bottom - baseline - 4),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
    return output
