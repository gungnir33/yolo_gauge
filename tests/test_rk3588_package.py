from __future__ import annotations

import os
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_rk3588_package.py"

EXPECTED_FILES = {
    "MIGRATION_GUIDE.md",
    "SHA256SUMS",
    "requirements-rk3588.txt",
    "configs/rk3588.yaml",
    "models/yoloe-26s-rk3588-fp16.rknn",
    "models/yoloe-26s-rk3588-fp16.json",
    "scripts/run_rknn_detection.sh",
    "src/gauge_detector/__init__.py",
    "src/gauge_detector/backends.py",
    "src/gauge_detector/board_cli.py",
    "src/gauge_detector/config.py",
    "src/gauge_detector/crop.py",
    "src/gauge_detector/detector.py",
    "src/gauge_detector/io_utils.py",
    "src/gauge_detector/postprocess.py",
    "src/gauge_detector/preprocess.py",
    "src/gauge_detector/runtime_output.py",
    "src/gauge_detector/types.py",
    "src/gauge_detector/visualization.py",
}


def _package_files(package: Path) -> set[str]:
    return {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}


def _write_metadata(model: Path, *, target: str = "rk3588") -> Path:
    metadata = model.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": target,
                "quantize": 16,
                "batch": 1,
                "toolkit_version": "2.3.2",
                "input_shape": [544, 960],
                "input_layout": "NHWC",
                "input_dtype": "uint8",
                "raw_head": "one2one",
                "raw_box_format": "xywh",
                "prompts": [
                    "analog gauge",
                    "dial gauge",
                    "pressure gauge",
                    "pressure meter",
                    "industrial gauge",
                ],
                "rknn_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return metadata


def test_builder_creates_self_contained_minimal_runtime_package(tmp_path):
    model = tmp_path / "input.rknn"
    model.write_bytes(b"fake-rknn-for-package-test")
    metadata = _write_metadata(model)
    package = tmp_path / "rk3588-runtime"
    archive = tmp_path / "rk3588-runtime.tar.gz"

    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model",
            str(model),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(package),
            "--archive",
            str(archive),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _package_files(package) == EXPECTED_FILES
    assert not any(path.is_symlink() for path in package.rglob("*"))
    config = yaml.safe_load((package / "configs/rk3588.yaml").read_text(encoding="utf-8"))
    assert config["model"]["rknn_path"] == "models/yoloe-26s-rk3588-fp16.rknn"
    requirements = (package / "requirements-rk3588.txt").read_text(encoding="utf-8").splitlines()
    assert requirements == [
        "# RKNNLite 必须安装与板端 Python ABI、librknnrt 和 RKNPU 驱动兼容的厂商 wheel。",
        "numpy==1.24.4",
        "opencv-python-headless==4.8.1.78",
        "PyYAML>=6,<7",
    ]

    checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=package,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checksum.returncode == 0, checksum.stderr
    assert len(checksum.stdout.splitlines()) == len(EXPECTED_FILES) - 1

    with tarfile.open(archive, "r:gz") as handle:
        members = {member.name for member in handle.getmembers() if member.isfile()}
    assert members == {f"{package.name}/{name}" for name in EXPECTED_FILES}


def test_packaged_board_cli_help_does_not_require_host_modules(tmp_path):
    model = tmp_path / "input.rknn"
    model.write_bytes(b"fake-rknn-for-package-test")
    metadata = _write_metadata(model)
    package = tmp_path / "rk3588-runtime"

    build = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model",
            str(model),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(package),
            "--no-archive",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    env = {**os.environ, "PYTHONPATH": str(package / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "gauge_detector.board_cli", "--help"],
        cwd=package,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "detect-dir" in result.stdout
    assert "convert-rknn" not in result.stdout


def test_builder_rejects_model_that_does_not_match_metadata(tmp_path):
    model = tmp_path / "input.rknn"
    model.write_bytes(b"original")
    metadata = _write_metadata(model)
    model.write_bytes(b"different-model")

    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model",
            str(model),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(tmp_path / "package"),
            "--no-archive",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SHA256" in result.stderr


def test_builder_rejects_archive_inside_output_directory(tmp_path):
    model = tmp_path / "input.rknn"
    model.write_bytes(b"model")
    metadata = _write_metadata(model)
    package = tmp_path / "package"

    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model",
            str(model),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(package),
            "--archive",
            str(package / "self.tar.gz"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "压缩包" in result.stderr
