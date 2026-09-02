from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Any


@dataclass(frozen=True)
class RKNNBuildConfig:
    target: str = "rk3588"
    quantize: int = 16
    batch: int = 1

    def __post_init__(self) -> None:
        if self.target.lower() != "rk3588":
            raise ValueError("Only target rk3588 is supported.")
        if self.quantize not in {8, 16}:
            raise ValueError("quantize must be 8 or 16")
        if self.batch != 1:
            raise ValueError("Only batch=1 is supported.")


def write_calibration_list(images: Iterable[str | Path], output: str | Path) -> Path:
    paths = sorted({Path(path).resolve() for path in images}, key=lambda path: str(path))
    if not paths:
        raise ValueError("Calibration image list must not be empty.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration image not found: {missing[0]}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    return destination


def _onnx_contract(path: str | Path) -> tuple[list[list[int | str]], list[list[int | str]]]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("RKNN conversion validation requires the optional 'onnx' package.") from exc
    model = onnx.load(str(path), load_external_data=False)
    def shapes(values) -> list[list[int | str]]:
        result: list[list[int | str]] = []
        for value in values:
            dimensions: list[int | str] = []
            for dimension in value.type.tensor_type.shape.dim:
                dimensions.append(
                    dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or "?"
                )
            result.append(dimensions)
        return result

    return shapes(model.graph.input), shapes(model.graph.output)


def _load_source_metadata(source: Path) -> dict[str, Any]:
    metadata_path = source.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"RKNN source metadata not found: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RKNN source metadata root must be an object.")
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if payload.get("onnx_sha256") != actual_sha:
        raise ValueError("RKNN source ONNX SHA256 does not match its metadata.")
    if payload.get("end2end_output") != "raw" or payload.get("raw_head") != "one2one":
        raise ValueError("RKNN source metadata must identify a one2one raw output head.")
    if payload.get("raw_box_format") != "xywh":
        raise ValueError("RKNN source metadata must use raw_box_format=xywh.")
    if payload.get("input_shape") != [544, 960]:
        raise ValueError("RKNN source metadata must use input_shape=[544, 960].")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts or any(not isinstance(item, str) or not item for item in prompts):
        raise ValueError("RKNN source metadata must contain a non-empty prompt list.")
    return payload


def _default_rknn_factory():
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "RKNN conversion requires rknn-toolkit2 in the dedicated Python 3.10 environment."
        ) from exc
    return RKNN(verbose=False)


def _toolkit_version() -> str:
    try:
        return importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _check_return_code(operation: str, code: Any) -> None:
    if type(code) is not int or code != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {code}")


def convert_onnx_to_rknn(
    onnx_path: str | Path,
    output_path: str | Path,
    config: RKNNBuildConfig,
    dataset: str | Path | None = None,
    *,
    rknn_factory: Callable[[], Any] | None = None,
) -> Path:
    if config.quantize == 8 and dataset is None:
        raise ValueError("INT8 conversion requires a calibration dataset list.")
    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    dataset_path = Path(dataset) if dataset is not None else None
    if dataset_path is not None and not dataset_path.is_file():
        raise FileNotFoundError(f"Calibration dataset list not found: {dataset_path}")
    source_metadata = _load_source_metadata(source)
    input_shapes, output_shapes = _onnx_contract(source)
    if input_shapes != [[1, 3, 544, 960]]:
        raise ValueError(f"RKNN source requires one static input [1, 3, 544, 960]; got {input_shapes}")
    if any(len(shape) == 3 and shape[-1] == 6 for shape in output_shapes):
        raise ValueError("RKNN conversion requires raw output ONNX; end-to-end 1xNx6 output is unsupported.")
    expected_channels = 4 + len(source_metadata["prompts"])
    if (
        len(output_shapes) != 1
        or len(output_shapes[0]) != 3
        or output_shapes[0][0] != 1
        or output_shapes[0][1] != expected_channels
        or not isinstance(output_shapes[0][2], int)
        or output_shapes[0][2] <= 0
    ):
        raise ValueError(
            f"RKNN source requires one static raw output [1, {expected_channels}, anchors]; got {output_shapes}"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime = (rknn_factory or _default_rknn_factory)()
    try:
        _check_return_code(
            "config",
            runtime.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=config.target.lower(),
            ),
        )
        _check_return_code("load_onnx", runtime.load_onnx(model=str(source)))
        build_options: dict[str, Any] = {
            "do_quantization": config.quantize == 8,
            "rknn_batch_size": config.batch,
        }
        if dataset_path is not None:
            build_options["dataset"] = str(dataset_path)
        _check_return_code("build", runtime.build(**build_options))
        _check_return_code("export_rknn", runtime.export_rknn(str(destination)))
    finally:
        runtime.release()
    if not destination.is_file():
        raise RuntimeError(f"RKNN export did not create output: {destination}")
    metadata = {
        "schema_version": 1,
        "target": config.target.lower(),
        "quantize": config.quantize,
        "batch": config.batch,
        "source_onnx": source.name,
        "source_onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(source.with_suffix(".json").read_bytes()).hexdigest(),
        "source_output_shapes": output_shapes,
        "toolkit_version": _toolkit_version(),
        "input_shape": source_metadata["input_shape"],
        "input_layout": "NHWC",
        "input_dtype": "uint8",
        "raw_head": source_metadata["raw_head"],
        "raw_box_format": source_metadata["raw_box_format"],
        "prompts": source_metadata["prompts"],
        "normalization": {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]]},
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "rknn_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
