import json

import pytest

from gauge_detector.prompt_profile import (
    PromptProfileMetadata,
    load_profile_metadata,
    prepare_prompt_profile,
    save_profile_metadata,
    validate_profile,
)


def test_profile_validation_rejects_changed_prompt_order(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    metadata = PromptProfileMetadata.create(checkpoint, ["dial gauge", "pressure gauge"], 960)

    with pytest.raises(ValueError, match="prompt"):
        validate_profile(metadata, checkpoint, ["pressure gauge", "dial gauge"], 960)


def test_profile_validation_rejects_changed_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-a")
    metadata = PromptProfileMetadata.create(checkpoint, ["dial gauge"], 960)
    checkpoint.write_bytes(b"weights-b")

    with pytest.raises(ValueError, match="checkpoint"):
        validate_profile(metadata, checkpoint, ["dial gauge"], 960)


def test_profile_metadata_round_trip_is_deterministic(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    metadata = PromptProfileMetadata.create(checkpoint, ["dial gauge", "pressure gauge"], 960)
    output = tmp_path / "gauge-prompts.json"

    save_profile_metadata(output, metadata)

    assert load_profile_metadata(output) == metadata
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize("prompts", [[], [""], ["dial gauge", "dial gauge"]])
def test_profile_rejects_empty_or_duplicate_prompts(tmp_path, prompts):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(ValueError, match="prompt"):
        PromptProfileMetadata.create(checkpoint, prompts, 960)


def test_model_saves_initialized_prompt_embeddings(tmp_path):
    from gauge_detector.model import YOLOEModel

    class FakeRawModel:
        def __init__(self):
            self.saved = []

        def set_classes(self, prompts):
            self.prompts = prompts

        def save_prompt_embeddings(self, path):
            self.saved.append(path)
            return path

    output = tmp_path / "gauge-prompts.npz"
    model = YOLOEModel("yoloe-26s-seg.pt", device="cpu", half=False)
    model._model = FakeRawModel()
    model.set_text_prompts(["dial gauge"])

    assert model.save_prompt_embeddings(output) == output
    assert model.raw.saved == [output]


def test_model_loads_prompt_embeddings_without_text_encoder(tmp_path):
    from gauge_detector.model import YOLOEModel

    class FakeRawModel:
        def __init__(self):
            self.loaded = []

        def load_prompt_embeddings(self, path):
            self.loaded.append(path)

    profile = tmp_path / "gauge-prompts.npz"
    profile.write_bytes(b"npz")
    model = YOLOEModel("yoloe-26s-seg.pt", device="cpu", half=False)
    model._model = FakeRawModel()

    model.load_prompt_embeddings(profile, ["dial gauge", "pressure gauge"])

    assert model.raw.loaded == [profile]
    assert model.text_prompts == ("dial gauge", "pressure gauge")


def test_prepare_prompt_profile_writes_embeddings_and_valid_metadata(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model:\n  name: {checkpoint}\n  device: cpu\n  imgsz: 960\n  half: false\n"
        "text_prompt:\n  prompts: [dial gauge, pressure gauge]\n",
        encoding="utf-8",
    )

    class FakeModel:
        def __init__(self, *args):
            self.prompts = None

        def set_text_prompts(self, prompts):
            self.prompts = prompts

        def save_prompt_embeddings(self, path):
            path.write_bytes(b"static-embeddings")
            return path

    output = tmp_path / "artifacts" / "gauge-prompts.npz"
    profile, metadata_path = prepare_prompt_profile(config, output, model_factory=FakeModel)

    assert profile.read_bytes() == b"static-embeddings"
    metadata = load_profile_metadata(metadata_path)
    assert metadata.prompts == ("dial gauge", "pressure gauge")
    assert metadata.checkpoint_sha256 == "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
