import numpy as np

from gauge_detector.types import Detection
from gauge_detector.visualization import BOX_COLORS, draw_detections


def test_draws_green_box_without_changing_input():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    output = draw_detections(image, [Detection(0, "instrument", 0.8, 10, 20, 50, 70)], thickness=1)
    assert tuple(output[20, 10]) == (0, 255, 0)
    assert tuple(output[0, 0]) == (0, 0, 0)
    assert not image.any()


def test_each_box_uses_next_color_and_draws_number():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    detections = [
        Detection(0, "instrument", 0.8, 20, 40, 60, 90),
        Detection(0, "instrument", 0.7, 90, 40, 140, 90),
    ]
    output = draw_detections(image, detections, thickness=1)
    assert tuple(output[40, 20]) == BOX_COLORS[0]
    assert tuple(output[40, 90]) == BOX_COLORS[1]
    # Colored label backgrounds are placed above both boxes.
    assert tuple(output[15, 20]) == BOX_COLORS[0]
    assert tuple(output[15, 90]) == BOX_COLORS[1]
