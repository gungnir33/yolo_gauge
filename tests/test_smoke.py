import os

import numpy as np
import pytest


def test_cli_parser_loads():
    from gauge_detector.cli import build_parser

    parser = build_parser()
    assert parser.prog == "python -m gauge_detector"
    args = parser.parse_args(["detect", "--image", "image.jpg"])
    assert not hasattr(args, "profile")

    profile_args = parser.parse_args(
        ["prepare-profile", "--config", "configs/default.yaml", "--output", "artifacts/gauge-prompts.npz"]
    )
    assert profile_args.command == "prepare-profile"
    assert profile_args.output == "artifacts/gauge-prompts.npz"


def test_prompt_free_checkpoint_is_rejected():
    from gauge_detector.model import YOLOEModel

    with pytest.raises(ValueError, match="prompt-free"):
        YOLOEModel("yoloe-26s-seg-pf.pt", device="cpu")


def test_text_prompts_are_initialized_once():
    from gauge_detector.model import YOLOEModel

    class FakeRawModel:
        def __init__(self):
            self.calls = []

        def set_classes(self, prompts):
            self.calls.append(prompts)

    model = YOLOEModel("yoloe-26s-seg.pt", device="cpu", half=False)
    model._model = FakeRawModel()
    model.set_text_prompts(["dial gauge", "digital meter"])
    model.set_text_prompts(["dial gauge", "digital meter"])
    assert model.raw.calls == [["dial gauge", "digital meter"]]


def test_detector_uses_agnostic_nms_and_unifies_class(monkeypatch, tmp_path):
    import torch

    from gauge_detector.backends import PyTorchBackend
    from gauge_detector.detector import GaugeDetector

    class Boxes:
        xyxy = torch.tensor([[10.0, 20.0, 50.0, 60.0], [11.0, 21.0, 51.0, 61.0]])
        conf = torch.tensor([0.9, 0.8])

        def __len__(self):
            return 2

    class Result:
        boxes = Boxes()

    class FakeModel:
        instances = []

        def __init__(self, *args):
            self.prompt_calls = []
            self.predict_options = None
            self.imgsz = args[2]
            FakeModel.instances.append(self)

        def set_text_prompts(self, prompts):
            self.prompt_calls.append(prompts)

        def warmup(self, runs):
            pass

        def predict(self, image, **kwargs):
            self.predict_options = kwargs
            return [Result()]

    config = tmp_path / "config.yaml"
    config.write_text(
        "text_prompt:\n"
        "  prompts: [dial gauge, digital meter]\n"
        "  unified_class_name: instrument\n"
        "detection:\n"
        "  conf: 0.25\n"
        "  iou: 0.50\n"
        "  agnostic_nms: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gauge_detector.detector.create_backend",
        lambda cfg: PyTorchBackend(
            cfg["model"]["name"],
            cfg["model"]["device"],
            cfg["model"]["imgsz"],
            cfg["model"]["half"],
            cfg["text_prompt"]["prompts"],
            model_factory=FakeModel,
        ),
    )
    detector = GaugeDetector(str(config))
    result = detector.detect_array(np.zeros((100, 100, 3), dtype=np.uint8))
    model = FakeModel.instances[0]
    assert model.prompt_calls == [["dial gauge", "digital meter"]]
    assert model.predict_options["agnostic_nms"] is True
    assert model.predict_options["iou"] == 0.5
    assert len(result.detections) == 1
    assert result.detections[0].class_id == 0
    assert result.detections[0].class_name == "instrument"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("YOLOE_RUN_INTEGRATION"), reason="set YOLOE_RUN_INTEGRATION=1 to load a checkpoint")
def test_yoloe_import_and_model_load():
    from gauge_detector.model import YOLOEModel

    model = YOLOEModel("yoloe-26n-seg.pt", device="cpu", imgsz=640, half=False)
    assert model.load() is model.raw
