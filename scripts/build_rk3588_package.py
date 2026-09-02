from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "yoloe-26s-rk3588-fp16.rknn"
RUNTIME_MODULES = (
    "__init__.py",
    "backends.py",
    "board_cli.py",
    "config.py",
    "crop.py",
    "detector.py",
    "io_utils.py",
    "postprocess.py",
    "preprocess.py",
    "runtime_output.py",
    "types.py",
    "visualization.py",
)
EXPECTED_CONTRACT = {
    "schema_version": 1,
    "target": "rk3588",
    "quantize": 16,
    "batch": 1,
    "toolkit_version": "2.3.2",
    "input_shape": [544, 960],
    "input_layout": "NHWC",
    "input_dtype": "uint8",
    "raw_head": "one2one",
    "raw_box_format": "xywh",
}


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_checksums(package: Path) -> None:
    lines = []
    for path in sorted(item for item in package.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(package)}")
    (package / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_model(model: Path, metadata_path: Path, expected_prompts: list[str]) -> dict:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"RKNN 模型元数据不存在: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("RKNN 模型元数据根节点必须是对象。")
    for key, expected in EXPECTED_CONTRACT.items():
        if metadata.get(key) != expected:
            raise ValueError(f"RKNN 模型元数据契约不匹配: {key} 应为 {expected!r}")
    if metadata.get("prompts") != expected_prompts:
        raise ValueError("RKNN 模型元数据中的 prompts 与 RK3588 配置不一致。")
    actual_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    if metadata.get("rknn_sha256") != actual_sha:
        raise ValueError("RKNN 模型 SHA256 与元数据不一致。")
    return metadata


def build_package(model: Path, metadata_path: Path, output_dir: Path, archive: Path | None) -> None:
    if not model.is_file():
        raise FileNotFoundError(f"RKNN 模型不存在: {model}")
    if archive is not None and archive.is_relative_to(output_dir):
        raise ValueError("压缩包不能位于待打包的输出目录内部。")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请先移走或改用其他路径: {output_dir}")
    if archive is not None and archive.exists():
        raise FileExistsError(f"压缩包已存在，请先移走或改用其他路径: {archive}")

    with (PROJECT_ROOT / "configs/rk3588.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    _validate_model(model, metadata_path, config["text_prompt"]["prompts"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rk3588-package-", dir=output_dir.parent) as temp:
        staging = Path(temp) / output_dir.name
        _copy(PROJECT_ROOT / "docs" / "RK3588_MIGRATION_GUIDE.md", staging / "MIGRATION_GUIDE.md")
        _copy(PROJECT_ROOT / "requirements-rk3588.txt", staging / "requirements-rk3588.txt")
        _copy(PROJECT_ROOT / "scripts" / "run_rknn_detection.sh", staging / "scripts/run_rknn_detection.sh")
        _copy(model, staging / "models" / MODEL_NAME)
        _copy(metadata_path, staging / "models" / Path(MODEL_NAME).with_suffix(".json").name)
        for module in RUNTIME_MODULES:
            _copy(PROJECT_ROOT / "src" / "gauge_detector" / module, staging / "src/gauge_detector" / module)

        config["model"]["rknn_path"] = f"models/{MODEL_NAME}"
        config_path = staging / "configs/rk3588.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        _write_checksums(staging)
        staging.rename(output_dir)

    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output_dir, arcname=output_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成最小 RK3588 Python 仪表检测迁移包")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rk3588" / MODEL_NAME,
        help="已转换的 RKNN 模型",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dist/yolo_gauge_rk3588_python",
        help="迁移目录",
    )
    parser.add_argument("--metadata", type=Path, help="RKNN 模型元数据；默认使用模型同名 JSON")
    parser.add_argument("--archive", type=Path, help="tar.gz 输出路径；默认紧邻迁移目录")
    parser.add_argument("--no-archive", action="store_true", help="只生成目录，不生成 tar.gz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    model = args.model.resolve()
    metadata = (args.metadata or model.with_suffix(".json")).resolve()
    archive = None if args.no_archive else (args.archive or Path(f"{output_dir}.tar.gz")).resolve()
    build_package(model, metadata, output_dir, archive)
    print(f"迁移目录: {output_dir}")
    if archive is not None:
        print(f"压缩包: {archive}")


if __name__ == "__main__":
    main()
