import numpy as np
import pytest

from gauge_detector.preprocess import LetterboxTransform, letterbox_rgb, onnx_tensor, restore_xyxy


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

