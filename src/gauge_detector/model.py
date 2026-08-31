from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class YOLOEModel:
    """Lazy, single-instance wrapper around an Ultralytics YOLOE model."""

    def __init__(self, model_name: str, device: str = "auto", imgsz: int = 960, half: bool = True):
        if "-pf" in Path(model_name).stem:
            raise ValueError(
                "Text Prompt mode requires a normal YOLOE *-seg.pt model, "
                "not a *-seg-pf.pt prompt-free model."
            )
        self.model_name = model_name
        self.imgsz = int(imgsz)
        self.device, cuda_available = self._resolve_device(device)
        self.half = bool(half and cuda_available and self.device != "cpu")
        self._model: Any = None
        self._text_prompts: tuple[str, ...] | None = None

    @staticmethod
    def _resolve_device(requested: str | int) -> tuple[str | int, bool]:
        import torch

        cuda = torch.cuda.is_available()
        if str(requested).lower() == "auto":
            if cuda:
                return 0, True
            LOGGER.info("CUDA unavailable. Falling back to CPU.")
            return "cpu", False
        if str(requested).lower() != "cpu" and not cuda:
            LOGGER.warning("CUDA unavailable. Falling back to CPU.")
            return "cpu", False
        return requested, cuda

    @property
    def raw(self):
        return self.load()

    def load(self):
        if self._model is None:
            from ultralytics import YOLOE

            LOGGER.info("Loading model=%s device=%s imgsz=%s", self.model_name, self.device, self.imgsz)
            self._model = YOLOE(self.model_name)
        return self._model

    def predict(self, source, **kwargs):
        options = {
            "conf": kwargs.pop("conf", 0.15),
            "imgsz": kwargs.pop("imgsz", self.imgsz),
            "device": kwargs.pop("device", self.device),
            "verbose": kwargs.pop("verbose", False),
        }
        if kwargs.pop("half", self.half):
            options["half"] = True
        options.update(kwargs)
        return self.raw.predict(source=source, **options)

    def set_text_prompts(self, prompts: list[str]) -> None:
        normalized = tuple(prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip())
        if not normalized:
            raise ValueError("At least one YOLOE text prompt is required.")
        if len(normalized) != len(prompts) or len(set(normalized)) != len(normalized):
            raise ValueError("YOLOE text prompts must be non-empty and unique.")
        if self._text_prompts is not None:
            if self._text_prompts != normalized:
                raise RuntimeError("YOLOE text prompts have already been initialized for this model instance.")
            return
        try:
            self.raw.set_classes(list(normalized))
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize YOLOE text prompts. Check the Ultralytics version and whether "
                "the YOLOE text encoder resources are installed and available."
            ) from exc
        self._text_prompts = normalized
        LOGGER.info("Text Prompt initialized: %d prompt(s)", len(normalized))

    def warmup(self, runs: int = 3) -> None:
        if runs <= 0:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy, conf=0.99, max_det=1)
