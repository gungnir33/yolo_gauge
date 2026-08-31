"""YOLOE text-prompt industrial instrument detector."""

from .detector import GaugeDetector
from .types import Detection, DetectionResult

__all__ = ["Detection", "DetectionResult", "GaugeDetector"]
__version__ = "0.1.0"
