from __future__ import annotations

from .config import load_config
from .model import YOLOEModel


def export_model(config_path: str, output_format: str):
    if output_format not in {"engine", "onnx"}:
        raise ValueError("Export format must be 'engine' or 'onnx'.")
    config = load_config(config_path)
    cfg = config["model"]
    model = YOLOEModel(cfg["name"], cfg["device"], cfg["imgsz"], cfg["half"])
    model.set_text_prompts(config["text_prompt"]["prompts"])
    return model.raw.export(format=output_format, imgsz=cfg["imgsz"], device=model.device, half=model.half)
