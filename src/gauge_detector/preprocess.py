from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    ratio: float
    pad_x: float
    pad_y: float
    input_size: int


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR ndarray with shape HxWx3")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")


def letterbox_rgb(
    image: np.ndarray,
    size: int,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxTransform]:
    _validate_image(image)
    input_size = int(size)
    if input_size <= 0:
        raise ValueError("letterbox size must be positive")
    height, width = image.shape[:2]
    ratio = min(input_size / height, input_size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if (resized_width, resized_height) != (width, height):
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    horizontal = input_size - resized_width
    vertical = input_size - resized_height
    left = round(horizontal / 2 - 0.1)
    right = round(horizontal / 2 + 0.1)
    top = round(vertical / 2 - 0.1)
    bottom = round(vertical / 2 + 0.1)
    rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return np.ascontiguousarray(rgb), LetterboxTransform(ratio, float(left), float(top), input_size)


def onnx_tensor(
    image: np.ndarray,
    size: int,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxTransform]:
    rgb, transform = letterbox_rgb(image, size, pad_color)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    return tensor, transform


def restore_xyxy(
    box: list[float] | tuple[float, ...] | np.ndarray,
    transform: LetterboxTransform,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[float, float, float, float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = min(float(width), max(0.0, (float(box[0]) - transform.pad_x) / transform.ratio))
    y1 = min(float(height), max(0.0, (float(box[1]) - transform.pad_y) / transform.ratio))
    x2 = min(float(width), max(0.0, (float(box[2]) - transform.pad_x) / transform.ratio))
    y2 = min(float(height), max(0.0, (float(box[3]) - transform.pad_y) / transform.ratio))
    return x1, y1, x2, y2
