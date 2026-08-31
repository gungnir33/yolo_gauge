from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {"name": "yoloe-26s-seg.pt", "device": "auto", "imgsz": 960, "half": True},
    "text_prompt": {"prompts": [], "unified_class_name": "instrument"},
    "detection": {"conf": 0.25, "iou": 0.50, "agnostic_nms": True, "max_det": 20},
    "postprocess": {
        "single_target": {"enabled": True, "containment_threshold": 0.90},
        "geometry_filter": {
            "enabled": False,
            "min_area_ratio": 0.0003,
            "max_area_ratio": 0.5,
            "min_aspect_ratio": 0.45,
            "max_aspect_ratio": 2.2,
        },
    },
    "visualization": {
        "box_color": {"b": 0, "g": 255, "r": 0},
        "thickness": 3,
        "draw_label": False,
        "draw_confidence": False,
    },
    "output": {"save_json": True, "save_annotated_image": True, "save_crops": True},
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return _merge(config, loaded)
