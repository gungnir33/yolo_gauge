from __future__ import annotations

import hashlib
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


def _onnx_output_shapes(path: str | Path) -> list[list[int | str]]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("RKNN conversion validation requires the optional 'onnx' package.") from exc
    model = onnx.load(str(path), load_external_data=False)
    shapes: list[list[int | str]] = []
    for output in model.graph.output:
        dimensions: list[int | str] = []
        for dimension in output.type.tensor_type.shape.dim:
            dimensions.append(dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or "?")
        shapes.append(dimensions)
    return shapes


def _default_rknn_factory():
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "RKNN conversion requires rknn-toolkit2 in the dedicated Python 3.10 environment."
        ) from exc
    return RKNN(verbose=False)


def _check_return_code(operation: str, code: Any) -> None:
    if code not in (None, 0):
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
    output_shapes = _onnx_output_shapes(source)
    if any(len(shape) == 3 and shape[-1] == 6 for shape in output_shapes):
        raise ValueError("RKNN conversion requires raw output ONNX; end-to-end 1xNx6 output is unsupported.")
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
        "source_output_shapes": output_shapes,
        "normalization": {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]]},
        "dataset": str(dataset_path) if dataset_path is not None else None,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
