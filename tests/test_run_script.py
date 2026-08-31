from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_detection.sh"


def run_script(*args: str, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def make_fake_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / "configs").mkdir()
    shutil.copy2(SCRIPT, project / "scripts" / "run_detection.sh")
    (project / "configs" / "default.yaml").write_text("model: {}\n", encoding="utf-8")
    capture = project / "captured.txt"
    fake_python = project / ".venv" / "bin" / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    return project, capture


def test_one_click_script_help():
    result = run_script("--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout


def test_one_click_script_routes_directory_to_detect_dir(tmp_path):
    project, capture = make_fake_project(tmp_path)
    input_dir = project / "输入 图片"
    input_dir.mkdir()
    output_dir = project / "输出 结果"
    env = {**os.environ, "CAPTURE_PATH": str(capture)}

    result = subprocess.run(
        [
            "bash",
            str(project / "scripts" / "run_detection.sh"),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--recursive",
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "gauge_detector",
        "detect-dir",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--config",
        str(project / "configs" / "default.yaml"),
        "--recursive",
    ]


def test_one_click_script_routes_image_to_detect(tmp_path):
    project, capture = make_fake_project(tmp_path)
    image = project / "测试 图片.jpg"
    image.write_bytes(b"not decoded by fake runner")
    output_dir = project / "results"
    env = {**os.environ, "CAPTURE_PATH": str(capture)}

    result = subprocess.run(
        [
            "bash",
            str(project / "scripts" / "run_detection.sh"),
            "--input",
            str(image),
            "--output",
            str(output_dir),
        ],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "gauge_detector",
        "detect",
        "--image",
        str(image),
        "--output",
        str(output_dir),
        "--config",
        str(project / "configs" / "default.yaml"),
    ]


def test_one_click_script_rejects_missing_input(tmp_path):
    result = run_script("--input", str(tmp_path / "missing"))
    assert result.returncode == 2
    assert "输入路径不存在" in result.stderr


def test_rknn_launch_script_help():
    script = Path("scripts/run_rknn_detection.sh")

    result = subprocess.run(["bash", str(script), "--help"], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--core-mask" in result.stdout
