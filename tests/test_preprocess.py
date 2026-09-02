import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gauge_detector.preprocess import LetterboxTransform, letterbox_rgb, onnx_tensor, restore_xyxy


PROJECT_ROOT = Path(__file__).parents[1]


def test_runtime_import_does_not_require_typing_typealias():
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
                "; ".join(
                    [
                        "import dataclasses, importlib.util, sys, types, typing",
                        "compat_typing = types.ModuleType('typing')",
                        "compat_typing.__dict__.update({k: v for k, v in typing.__dict__.items() if k != 'TypeAlias'})",
                        "sys.modules['typing'] = compat_typing",
                    "sys.modules['cv2'] = types.ModuleType('cv2')",
                    "sys.modules['numpy'] = types.ModuleType('numpy')",
                    "path = 'src/gauge_detector/preprocess.py'",
                    "spec = importlib.util.spec_from_file_location('preprocess_compat', path)",
                    "module = importlib.util.module_from_spec(spec)",
                    "sys.modules[spec.name] = module",
                    "spec.loader.exec_module(module)",
                    "print('import-ok')",
                ]
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-ok"


def test_letterbox_records_center_padding_for_wide_image_and_converts_rgb():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]

    rgb, transform = letterbox_rgb(image, 200, (114, 114, 114))

    assert rgb.shape == (200, 200, 3)
    assert rgb[50, 0].tolist() == [3, 2, 1]
    assert transform == LetterboxTransform(ratio=1.0, pad_x=0.0, pad_y=50.0, input_size=200)


def test_restore_xyxy_reverses_padding_and_scale():
    transform = LetterboxTransform(ratio=0.5, pad_x=10.0, pad_y=20.0, input_size=200)

    assert restore_xyxy([20, 30, 70, 80], transform, (200, 300)) == pytest.approx((20, 20, 120, 120))


def test_onnx_tensor_is_normalized_nchw_float32():
    image = np.full((20, 10, 3), 255, dtype=np.uint8)

    tensor, transform = onnx_tensor(image, 32, (114, 114, 114))

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert tensor.max() == 1.0
    assert transform.input_size == 32


def test_onnx_tensor_accepts_static_rectangular_input_shape():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    tensor, transform = onnx_tensor(image, (544, 960), (114, 114, 114))

    assert tensor.shape == (1, 3, 544, 960)
    assert transform.ratio == pytest.approx(0.5)
    assert transform.pad_x == 0
    assert transform.pad_y == 2
    assert transform.input_size == (544, 960)
