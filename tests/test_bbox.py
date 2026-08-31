import pytest

from gauge_detector.types import Detection


def test_bbox_properties():
    detection = Detection(0, "instrument", 0.9, 10, 20, 40, 60)
    assert detection.width == 30
    assert detection.height == 40
    assert detection.center == (25, 40)
    assert detection.xyxy == [10, 20, 40, 60]
    assert detection.xywh == [10, 20, 30, 40]


@pytest.mark.parametrize("box", [[-1, 0, 2, 2], [0, -1, 2, 2], [2, 2, 1, 3], [1, 1, 1, 2]])
def test_invalid_detection(box):
    with pytest.raises(ValueError):
        Detection(0, "instrument", 0.5, *box)
