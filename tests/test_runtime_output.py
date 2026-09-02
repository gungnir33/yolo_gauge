import numpy as np
import pytest

from gauge_detector.preprocess import LetterboxTransform
from gauge_detector.runtime_output import decode_end2end_output, decode_raw_output


def test_end2end_decoder_filters_confidence_and_restores_box():
    output = np.array([[[20, 30, 70, 80, 0.9, 3], [0, 0, 5, 5, 0.1, 0]]], dtype=np.float32)
    transform = LetterboxTransform(0.5, 10.0, 20.0, 200)

    detections = decode_end2end_output(output, transform, (200, 300), conf=0.15)

    assert len(detections) == 1
    assert detections[0].xyxy == pytest.approx([20, 20, 120, 120])
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].class_name == "instrument"
    assert detections[0].class_id == 0


def test_end2end_decoder_rejects_unknown_output_shape():
    transform = LetterboxTransform(1.0, 0.0, 0.0, 100)

    with pytest.raises(ValueError, match="1xNx6"):
        decode_end2end_output(np.zeros((1, 6, 10), dtype=np.float32), transform, (100, 100), 0.15)


def test_raw_decoder_uses_max_class_score_and_class_agnostic_nms():
    output = np.array(
        [
            [
                [50, 52, 150],
                [50, 52, 150],
                [40, 40, 20],
                [40, 40, 20],
                [0.9, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ],
        dtype=np.float32,
    )
    transform = LetterboxTransform(1.0, 0.0, 0.0, 200)

    detections = decode_raw_output(output, transform, (200, 200), conf=0.15, iou=0.5, max_det=20)

    assert len(detections) == 2
    assert detections[0].xyxy == pytest.approx([30, 30, 70, 70])
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[1].xyxy == pytest.approx([140, 140, 160, 160])
    assert detections[1].confidence == pytest.approx(0.7)


def test_raw_decoder_rejects_non_finite_values():
    output = np.zeros((1, 5, 1), dtype=np.float32)
    output[0, 4, 0] = np.nan
    transform = LetterboxTransform(1.0, 0.0, 0.0, 100)

    with pytest.raises(ValueError, match="finite"):
        decode_raw_output(output, transform, (100, 100), conf=0.15, iou=0.5, max_det=20)
