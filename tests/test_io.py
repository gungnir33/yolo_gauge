from pathlib import Path

from gauge_detector.config import load_config
from gauge_detector.io_utils import load_json, result_to_dict, save_json
from gauge_detector.types import Detection, DetectionResult


def test_json_roundtrip(tmp_path):
    path = save_json({"中文": [1, 2]}, tmp_path / "nested" / "test.json")
    assert load_json(path) == {"中文": [1, 2]}


def test_config_load(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("detection:\n  conf: 0.2\n", encoding="utf-8")
    config = load_config(path)
    assert config["detection"]["conf"] == 0.2
    assert config["model"]["name"] == "yoloe-26s-seg.pt"


def test_default_config_uses_focused_analog_gauge_profile():
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)
    assert config["text_prompt"]["prompts"] == [
        "analog gauge",
        "dial gauge",
        "pressure gauge",
        "pressure meter",
        "industrial gauge",
    ]
    assert config["detection"]["conf"] == 0.15
    assert config["model"]["imgsz"] == 960


def test_result_json_uses_integer_pixels_and_rounded_confidence():
    result = DetectionResult(
        "scene.jpg", 100, 50, [Detection(0, "instrument", 0.123456, 1.2, 2.2, 10.8, 20.8)], 12.345
    )
    data = result_to_dict(result)
    assert data["detections"][0]["confidence"] == 0.1235
    assert data["detections"][0]["bbox_xyxy"] == [1, 2, 11, 21]
    assert data["inference_ms"] == 12.35
    assert data["detections"][0]["class_name"] == "instrument"
