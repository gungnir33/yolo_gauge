from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Invalid bounding box.")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class DetectionResult:
    image_path: str
    image_width: int
    image_height: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

