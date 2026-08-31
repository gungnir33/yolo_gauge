from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .model import YOLOEModel
from .preprocess import InputShape, normalize_input_shape, onnx_tensor
from .runtime_output import decode_end2end_output
from .types import Detection


@dataclass(frozen=True)
class BackendPrediction:
    detections: list[Detection]
    inference_ms: float


class PyTorchBackend:
    def __init__(
        self,
        model_name: str,
        device: str,
        imgsz: int,
        half: bool,
        prompts: list[str],
        *,
        model_factory: Callable[..., Any] = YOLOEModel,
    ):
        self.model = model_factory(model_name, device, imgsz, half)
        self.model.set_text_prompts(prompts)

    @property
    def imgsz(self) -> int:
        return self.model.imgsz

    @imgsz.setter
    def imgsz(self, value: int) -> None:
        self.model.imgsz = int(value)

    def warmup(self, runs: int = 3) -> None:
        self.model.warmup(runs)

    def predict(self, image: np.ndarray, *, conf: float, iou: float, max_det: int) -> BackendPrediction:
        height, width = image.shape[:2]
        started = time.perf_counter()
        results = self.model.predict(image, conf=conf, iou=iou, agnostic_nms=True, max_det=max_det)
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections: list[Detection] = []
        result = results[0]
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidence = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(xyxy, confidence):
                x1, y1, x2, y2 = box.tolist()
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                if x2 > x1 and y2 > y1:
                    detections.append(Detection(0, "instrument", float(score), x1, y1, x2, y2))
        return BackendPrediction(detections, elapsed_ms)


class ONNXBackend:
    def __init__(
        self,
        model_path: str | Path,
        imgsz: InputShape,
        pad_color: tuple[int, int, int],
        *,
        session_factory: Callable[[str], Any] | None = None,
    ):
        self.model_path = Path(model_path)
        self.input_shape = normalize_input_shape(imgsz)
        self.imgsz = self.input_shape[0] if self.input_shape[0] == self.input_shape[1] else self.input_shape
        self.pad_color = tuple(int(value) for value in pad_color)
        if session_factory is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("ONNX backend requires the optional 'onnxruntime' package.") from exc
            session_factory = lambda path: ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.session = session_factory(str(self.model_path))
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        expected_input = [1, 3, *self.input_shape]
        if len(inputs) != 1 or list(inputs[0].shape) != expected_input or inputs[0].type != "tensor(float)":
            actual = [(item.name, item.shape, item.type) for item in inputs]
            raise ValueError(f"ONNX backend requires one static input {expected_input}; got {actual}")
        if len(outputs) != 1 or len(outputs[0].shape) != 3 or outputs[0].shape[0] != 1 or outputs[0].shape[2] != 6:
            actual = [(item.name, item.shape, item.type) for item in outputs]
            raise ValueError(f"ONNX backend requires one static 1xNx6 output; got {actual}")
        self.input_name = inputs[0].name

    def warmup(self, runs: int = 3) -> None:
        if runs <= 0:
            return
        dummy = np.zeros((*self.input_shape, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy, conf=0.99, iou=0.5, max_det=1)

    def predict(self, image: np.ndarray, *, conf: float, iou: float, max_det: int) -> BackendPrediction:
        tensor, transform = onnx_tensor(image, self.input_shape, self.pad_color)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = decode_end2end_output(outputs[0], transform, image.shape, conf)
        return BackendPrediction(detections[: int(max_det)], elapsed_ms)


def create_backend(
    config: dict,
    *,
    pytorch_model_factory: Callable[..., Any] = YOLOEModel,
    onnx_session_factory: Callable[[str], Any] | None = None,
):
    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "pytorch")).lower()
    if backend == "pytorch":
        return PyTorchBackend(
            model_config["name"],
            model_config.get("device", "auto"),
            model_config.get("imgsz", 960),
            model_config.get("half", True),
            config.get("text_prompt", {}).get("prompts", []),
            model_factory=pytorch_model_factory,
        )
    if backend == "onnx":
        return ONNXBackend(
            model_config["onnx_path"],
            model_config.get("input_shape", model_config.get("imgsz", 960)),
            tuple(model_config.get("pad_color", (114, 114, 114))),
            session_factory=onnx_session_factory,
        )
    raise ValueError(f"Unsupported model backend: {backend}")
