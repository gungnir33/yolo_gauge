from types import SimpleNamespace

import numpy as np
import pytest

from gauge_detector.backends import BackendPrediction, ONNXBackend, PyTorchBackend, create_backend
from gauge_detector.config import load_config
from gauge_detector.types import Detection


def _config(backend="pytorch"):
    return {
        "model": {
            "backend": backend,
            "name": "model.pt",
            "onnx_path": "model.onnx",
            "imgsz": 100,
            "device": "cpu",
            "half": False,
            "pad_color": [114, 114, 114],
        },
        "text_prompt": {"prompts": ["dial gauge"]},
    }


def test_default_config_keeps_pytorch_backend_and_portable_padding():
    config = load_config()

    assert config["model"]["backend"] == "pytorch"
    assert config["model"]["pad_color"] == [114, 114, 114]


def test_backend_factory_defaults_to_pytorch_with_injected_model():
    class FakeModel:
        def __init__(self, *args):
            self.imgsz = args[2]

        def set_text_prompts(self, prompts):
            self.prompts = prompts

    backend = create_backend(_config(), pytorch_model_factory=FakeModel)

    assert isinstance(backend, PyTorchBackend)
    assert backend.imgsz == 100


def test_backend_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        create_backend(_config("cuda-magic"))


def test_onnx_backend_initializes_session_once_and_decodes_output():
    class FakeSession:
        def __init__(self):
            self.run_calls = 0

        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 100, 100], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=[1, 300, 6], type="tensor(float)")]

        def run(self, output_names, feed):
            self.run_calls += 1
            assert feed["images"].shape == (1, 3, 100, 100)
            return [np.array([[[10, 20, 30, 40, 0.9, 2]]], dtype=np.float32)]

    session = FakeSession()
    backend = ONNXBackend("model.onnx", 100, (114, 114, 114), session_factory=lambda _: session)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    first = backend.predict(image, conf=0.15, iou=0.5, max_det=20)
    second = backend.predict(image, conf=0.15, iou=0.5, max_det=20)

    assert session.run_calls == 2
    assert len(first.detections) == 1
    assert first.detections[0].xyxy == pytest.approx([10, 20, 30, 40])
    assert second.inference_ms >= 0


def test_onnx_backend_supports_configured_rectangular_input():
    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 54, 96], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=[1, 300, 6], type="tensor(float)")]

        def run(self, output_names, feed):
            assert feed["images"].shape == (1, 3, 54, 96)
            return [np.empty((1, 0, 6), dtype=np.float32)]

    backend = ONNXBackend("model.onnx", (54, 96), (114, 114, 114), session_factory=lambda _: FakeSession())

    prediction = backend.predict(np.zeros((108, 192, 3), dtype=np.uint8), conf=0.15, iou=0.5, max_det=20)

    assert backend.input_shape == (54, 96)
    assert prediction.detections == []


def test_backend_factory_prefers_runtime_input_shape_for_onnx():
    config = _config("onnx")
    config["model"]["input_shape"] = [54, 96]

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 54, 96], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=[1, 300, 6], type="tensor(float)")]

    backend = create_backend(config, onnx_session_factory=lambda _: FakeSession())

    assert backend.input_shape == (54, 96)


def test_onnx_backend_rejects_dynamic_or_wrong_output_contract():
    class BadSession:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=["batch", 3, 100, 100], type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="output0", shape=[1, 6, 300], type="tensor(float)")]

    with pytest.raises(ValueError, match="static"):
        ONNXBackend("model.onnx", 100, (114, 114, 114), session_factory=lambda _: BadSession())


def test_gauge_detector_uses_backend_candidates_and_keeps_business_postprocess(monkeypatch, tmp_path):
    from gauge_detector.detector import GaugeDetector

    class FakeBackend:
        imgsz = 100

        def warmup(self, runs):
            self.warmup_runs = runs

        def predict(self, image, **kwargs):
            return BackendPrediction(
                [
                    Detection(0, "instrument", 0.9, 20, 20, 40, 40),
                    Detection(0, "instrument", 0.6, 0, 0, 80, 80),
                ],
                12.5,
            )

    backend = FakeBackend()
    monkeypatch.setattr("gauge_detector.detector.create_backend", lambda config: backend)
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  backend: onnx\n  name: ignored.pt\n  imgsz: 100\n"
        "text_prompt:\n  prompts: [dial gauge]\n"
        "detection:\n  conf: 0.15\n  iou: 0.5\n  agnostic_nms: true\n"
        "postprocess:\n  single_target:\n    enabled: true\n    containment_threshold: 0.9\n",
        encoding="utf-8",
    )

    detector = GaugeDetector(str(config), warmup_runs=2)
    result = detector.detect_array(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detector.model is backend
    assert backend.warmup_runs == 2
    assert result.inference_ms == 12.5
    assert result.detections[0].xyxy == [0, 0, 80, 80]
