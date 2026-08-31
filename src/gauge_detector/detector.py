from __future__ import annotations

import logging

import numpy as np

from .backends import create_backend
from .config import load_config
from .io_utils import read_image
from .postprocess import filter_geometry, remove_duplicate_boxes, select_single_target, sort_detections
from .types import Detection, DetectionResult

LOGGER = logging.getLogger(__name__)


class GaugeDetector:
    """YOLOE Text Prompt detector; the historical class name is kept for API compatibility."""

    def __init__(self, config_path: str = "configs/default.yaml", warmup_runs: int = 0):
        self.config = load_config(config_path)
        prompt_cfg = self.config["text_prompt"]
        prompts = prompt_cfg.get("prompts", [])
        if not isinstance(prompts, list):
            raise ValueError("text_prompt.prompts must be a list of strings.")
        self.text_prompts = prompts
        self.unified_class_name = str(prompt_cfg.get("unified_class_name", "instrument")).strip()
        if not self.unified_class_name:
            raise ValueError("text_prompt.unified_class_name must not be empty.")
        detection_cfg = self.config["detection"]
        if not bool(detection_cfg.get("agnostic_nms", True)):
            raise ValueError("Text Prompt Ensemble requires detection.agnostic_nms=true.")
        runtime_model = self.config["model"]["name"]
        self.model = create_backend(self.config)
        LOGGER.info(
            "Detector mode: YOLOE Text Prompt | Model: %s | Unified class: %s | Text prompts: %d | "
            "Confidence: %.2f | IoU: %.2f | Class-agnostic NMS: enabled",
            runtime_model,
            self.unified_class_name,
            len(self.text_prompts),
            float(detection_cfg["conf"]),
            float(detection_cfg["iou"]),
        )
        self.model.warmup(warmup_runs)

    def detect(self, image_path: str) -> DetectionResult:
        image = read_image(image_path)
        return self._detect_array(image, str(image_path))

    def detect_array(self, image: np.ndarray) -> DetectionResult:
        return self._detect_array(image, "<ndarray>")

    def _detect_array(self, image: np.ndarray, source_name: str) -> DetectionResult:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("image must be a non-empty BGR ndarray with shape HxWx3.")
        height, width = image.shape[:2]
        detection_cfg = self.config["detection"]
        prediction = self.model.predict(
            image,
            conf=detection_cfg["conf"],
            iou=detection_cfg["iou"],
            max_det=detection_cfg["max_det"],
        )
        elapsed_ms = prediction.inference_ms
        detections = [
            Detection(0, self.unified_class_name, item.confidence, item.x1, item.y1, item.x2, item.y2)
            for item in prediction.detections
        ]
        post_cfg = self.config["postprocess"]
        # Ultralytics receives agnostic_nms=True above. Apply the same class-independent
        # IoU threshold once more at the business boundary so duplicate Prompt boxes
        # cannot leak through version- or task-specific predictor behavior.
        detections = remove_duplicate_boxes(detections, float(detection_cfg["iou"]))
        detections = filter_geometry(detections, width, height, post_cfg["geometry_filter"])
        detections = select_single_target(detections, post_cfg["single_target"])
        detections = sort_detections(detections)
        LOGGER.info("input=%s detections=%d latency=%.2fms", source_name, len(detections), elapsed_ms)
        return DetectionResult(source_name, width, height, detections, elapsed_ms)
