import json
from pathlib import Path

import pytest

from gauge_detector.export import (
    build_detection_only_model,
    detection_yaml_for_checkpoint,
    export_detection_onnx,
    export_rknn_source_onnx,
    use_one2one_raw_output,
)
from gauge_detector.prompt_profile import PromptProfileMetadata, save_profile_metadata


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("yoloe-26n-seg.pt", "yoloe-26n.yaml"),
        ("yoloe-26s-seg.pt", "yoloe-26s.yaml"),
        ("/models/yoloe-11m-seg.pt", "yoloe-11m.yaml"),
    ],
)
def test_detection_yaml_matches_checkpoint_scale(checkpoint, expected):
    assert detection_yaml_for_checkpoint(checkpoint) == expected


def test_detection_yaml_rejects_prompt_free_checkpoint():
    with pytest.raises(ValueError, match="prompt-free"):
        detection_yaml_for_checkpoint("yoloe-26s-seg-pf.pt")


@pytest.mark.parametrize("checkpoint", ["yolo26s.pt", "yoloe-26s.pt", "custom-seg.pt"])
def test_detection_yaml_rejects_unsupported_checkpoint(checkpoint):
    with pytest.raises(ValueError, match="YOLOE segmentation"):
        detection_yaml_for_checkpoint(Path(checkpoint))


def _write_profile(tmp_path, checkpoint, prompts, imgsz=960):
    profile = tmp_path / "gauge-prompts.npz"
    profile.write_bytes(b"embeddings")
    metadata = PromptProfileMetadata.create(checkpoint, prompts, imgsz)
    save_profile_metadata(profile.with_suffix(".json"), metadata)
    return profile


def test_build_detection_only_model_loads_matching_architecture_and_profile(tmp_path):
    checkpoint = tmp_path / "yoloe-26s-seg.pt"
    checkpoint.write_bytes(b"weights")
    prompts = ["dial gauge", "pressure gauge"]
    profile = _write_profile(tmp_path, checkpoint, prompts)

    class FakeYOLOE:
        instances = []

        def __init__(self, architecture):
            self.architecture = architecture
            self.loaded_checkpoint = None
            self.loaded_profile = None
            self.instances.append(self)

        def load(self, checkpoint_path):
            self.loaded_checkpoint = checkpoint_path
            return self

        def load_prompt_embeddings(self, profile_path):
            self.loaded_profile = profile_path

    model = build_detection_only_model(
        checkpoint,
        profile,
        prompts,
        960,
        model_factory=FakeYOLOE,
    )

    assert model is FakeYOLOE.instances[0]
    assert model.architecture == "yoloe-26s.yaml"
    assert model.loaded_checkpoint == str(checkpoint)
    assert model.loaded_profile == profile


def test_export_detection_onnx_uses_static_batch_one_contract(tmp_path):
    checkpoint = tmp_path / "yoloe-26s-seg.pt"
    checkpoint.write_bytes(b"weights")
    prompts = ["dial gauge"]
    profile = _write_profile(tmp_path, checkpoint, prompts)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model:\n  name: {checkpoint}\n  device: cpu\n  imgsz: 960\n  half: false\n"
        "text_prompt:\n  prompts: [dial gauge]\n",
        encoding="utf-8",
    )

    class FakeYOLOE:
        instance = None

        def __init__(self, architecture):
            self.architecture = architecture
            self.export_options = None
            FakeYOLOE.instance = self

        def load(self, checkpoint_path):
            return self

        def load_prompt_embeddings(self, profile_path):
            self.profile_path = profile_path

        def export(self, **kwargs):
            self.export_options = kwargs
            exported = tmp_path / "ultralytics-output.onnx"
            exported.write_bytes(b"onnx")
            return str(exported)

    output = export_detection_onnx(config, profile, tmp_path / "exported", model_factory=FakeYOLOE)

    assert output == tmp_path / "exported" / "yoloe-26s.onnx"
    assert output.read_bytes() == b"onnx"
    assert FakeYOLOE.instance.export_options == {
        "format": "onnx",
        "imgsz": 960,
        "batch": 1,
        "dynamic": False,
        "opset": 19,
        "simplify": False,
        "nms": False,
        "agnostic_nms": True,
        "device": "cpu",
    }
    metadata = output.with_suffix(".json")
    assert metadata.is_file()


def test_export_detection_onnx_uses_runtime_input_shape_when_configured(tmp_path):
    checkpoint = tmp_path / "yoloe-26s-seg.pt"
    checkpoint.write_bytes(b"weights")
    prompts = ["dial gauge"]
    profile = _write_profile(tmp_path, checkpoint, prompts)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model:\n  name: {checkpoint}\n  imgsz: 960\n  input_shape: [544, 960]\n"
        "text_prompt:\n  prompts: [dial gauge]\n",
        encoding="utf-8",
    )

    class FakeYOLOE:
        instance = None

        def __init__(self, architecture):
            FakeYOLOE.instance = self

        def load(self, checkpoint_path):
            return self

        def load_prompt_embeddings(self, profile_path):
            pass

        def export(self, **kwargs):
            self.export_options = kwargs
            output = tmp_path / "runtime-shape.onnx"
            output.write_bytes(b"onnx")
            return output

    output = export_detection_onnx(config, profile, tmp_path / "exported", model_factory=FakeYOLOE)

    assert FakeYOLOE.instance.export_options["imgsz"] == (544, 960)
    assert '"input_shape": [\n    544,\n    960\n  ]' in output.with_suffix(".json").read_text(encoding="utf-8")


def test_export_rknn_source_onnx_disables_end2end_output(tmp_path):
    checkpoint = tmp_path / "yoloe-26s-seg.pt"
    checkpoint.write_bytes(b"weights")
    profile = _write_profile(tmp_path, checkpoint, ["dial gauge"])
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model:\n  name: {checkpoint}\n  imgsz: 960\n  input_shape: [544, 960]\n"
        "text_prompt:\n  prompts: [dial gauge]\n",
        encoding="utf-8",
    )

    class FakeYOLOE:
        instance = None

        def __init__(self, architecture):
            FakeYOLOE.instance = self

        def load(self, checkpoint_path):
            return self

        def load_prompt_embeddings(self, profile_path):
            pass

        def export(self, **kwargs):
            self.export_options = kwargs
            output = tmp_path / "raw.onnx"
            output.write_bytes(b"onnx")
            return output

    output = export_rknn_source_onnx(config, profile, tmp_path / "exported", model_factory=FakeYOLOE)

    assert FakeYOLOE.instance.export_options["end2end"] is False
    assert output.name == "yoloe-26s-rknn-source.onnx"
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["end2end_output"] == "raw"
    assert metadata["raw_head"] == "one2one"
    assert metadata["raw_box_format"] == "xywh"


def test_one2one_raw_output_reuses_nms_free_head_without_topk():
    one2one_box = object()
    one2one_class = object()
    one2one_contrastive = object()

    class Head:
        cv2 = object()
        cv3 = object()
        cv4 = object()
        one2one_cv2 = one2one_box
        one2one_cv3 = one2one_class
        one2one_cv4 = one2one_contrastive
        end2end = True

    head = Head()
    model = type("Wrapper", (), {"model": type("Network", (), {"model": [head]})()})()

    use_one2one_raw_output(model)

    assert head.cv2 is one2one_box
    assert head.cv3 is one2one_class
    assert head.cv4 is one2one_contrastive
    assert head.end2end is False
