from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .model import YOLOEModel
from .preprocess import normalize_input_shape
from .prompt_profile import load_profile_metadata, validate_profile


_YOLOE_SEGMENT_CHECKPOINT = re.compile(r"^(yoloe-(?:26[nsmxl]|11[sml]|v8[sml]))-seg$")


def detection_yaml_for_checkpoint(checkpoint: str | Path) -> str:
    stem = Path(checkpoint).stem
    if stem.endswith("-seg-pf"):
        raise ValueError("A prompt-free checkpoint cannot produce a text-prompt detection model.")
    match = _YOLOE_SEGMENT_CHECKPOINT.fullmatch(stem)
    if match is None:
        raise ValueError(f"Expected a supported YOLOE segmentation checkpoint, got: {checkpoint}")
    return f"{match.group(1)}.yaml"


def build_detection_only_model(
    checkpoint: str | Path,
    profile_path: str | Path,
    prompts: list[str],
    imgsz: int,
    *,
    model_factory: Callable[..., Any] | None = None,
):
    checkpoint_path = Path(checkpoint)
    profile = Path(profile_path)
    if not profile.is_file():
        raise FileNotFoundError(f"Prompt embedding profile not found: {profile}")
    metadata = load_profile_metadata(profile.with_suffix(".json"))
    validate_profile(metadata, checkpoint_path, prompts, imgsz)
    if model_factory is None:
        from ultralytics import YOLOE

        model_factory = YOLOE
    model = model_factory(detection_yaml_for_checkpoint(checkpoint_path)).load(str(checkpoint_path))
    model.load_prompt_embeddings(profile)
    return model


def use_one2one_raw_output(model: Any) -> None:
    """Expose the YOLO26 one-to-one head before export while leaving TopK to Python."""
    try:
        head = model.model.model[-1]
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Unable to locate the YOLO26 detection head for RKNN export.") from exc
    required = ("cv2", "cv3", "one2one_cv2", "one2one_cv3", "end2end")
    if any(not hasattr(head, name) for name in required):
        raise RuntimeError("RKNN source export requires a YOLO26 one-to-one detection head.")
    head.cv2 = head.one2one_cv2
    head.cv3 = head.one2one_cv3
    if hasattr(head, "cv4") and hasattr(head, "one2one_cv4"):
        head.cv4 = head.one2one_cv4
    head.end2end = False


def export_detection_onnx(
    config_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    *,
    model_factory: Callable[..., Any] | None = None,
) -> Path:
    return _export_static_onnx(config_path, profile_path, output_dir, model_factory=model_factory)


def export_rknn_source_onnx(
    config_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    *,
    model_factory: Callable[..., Any] | None = None,
) -> Path:
    return _export_static_onnx(
        config_path,
        profile_path,
        output_dir,
        model_factory=model_factory,
        end2end=False,
        destination_suffix="-rknn-source",
        output_contract="raw",
    )


def _export_static_onnx(
    config_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path,
    *,
    model_factory: Callable[..., Any] | None = None,
    end2end: bool | None = None,
    destination_suffix: str = "",
    output_contract: str = "1xNx6",
) -> Path:
    config = load_config(config_path)
    model_config = config["model"]
    checkpoint = Path(model_config["name"])
    prompts = config["text_prompt"]["prompts"]
    imgsz = int(model_config["imgsz"])
    configured_shape = model_config.get("input_shape")
    export_imgsz: int | tuple[int, int] = imgsz
    if configured_shape is not None:
        export_imgsz = normalize_input_shape(configured_shape)
    architecture = detection_yaml_for_checkpoint(checkpoint)
    model = build_detection_only_model(
        checkpoint,
        profile_path,
        prompts,
        imgsz,
        model_factory=model_factory,
    )
    if end2end is False and (model_factory is None or hasattr(model, "model")):
        use_one2one_raw_output(model)
    export_options: dict[str, Any] = {
        "format": "onnx",
        "imgsz": export_imgsz,
        "batch": 1,
        "dynamic": False,
        "opset": 19,
        "simplify": False,
        "nms": False,
        "agnostic_nms": True,
        "device": "cpu",
    }
    if end2end is not None:
        export_options["end2end"] = end2end
    exported_path = Path(model.export(**export_options))
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{Path(architecture).stem}{destination_suffix}.onnx"
    if exported_path.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()
        shutil.move(str(exported_path), destination)
    metadata = load_profile_metadata(Path(profile_path).with_suffix(".json"))
    export_metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "checkpoint_name": metadata.checkpoint_name,
        "checkpoint_sha256": metadata.checkpoint_sha256,
        "prompts": list(metadata.prompts),
        "imgsz": imgsz,
        "input_shape": list(normalize_input_shape(export_imgsz)),
        "batch": 1,
        "dynamic": False,
        "opset": 19,
        "agnostic_nms": True,
        "end2end_output": output_contract,
    }
    if output_contract == "raw":
        export_metadata["raw_head"] = "one2one"
        export_metadata["raw_box_format"] = "xywh"
    destination.with_suffix(".json").write_text(
        json.dumps(export_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def export_model(config_path: str, output_format: str):
    if output_format not in {"engine", "onnx"}:
        raise ValueError("Export format must be 'engine' or 'onnx'.")
    config = load_config(config_path)
    cfg = config["model"]
    model = YOLOEModel(cfg["name"], cfg["device"], cfg["imgsz"], cfg["half"])
    model.set_text_prompts(config["text_prompt"]["prompts"])
    return model.raw.export(format=output_format, imgsz=cfg["imgsz"], device=model.device, half=model.half)
