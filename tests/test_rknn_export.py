import json
import hashlib
from pathlib import Path

import pytest

from gauge_detector.rknn_export import RKNNBuildConfig, convert_onnx_to_rknn, write_calibration_list


def _write_source_metadata(path, *, raw_head="one2one", sha256=None):
    payload = {
        "end2end_output": "raw",
        "raw_head": raw_head,
        "raw_box_format": "xywh",
        "input_shape": [544, 960],
        "prompts": ["a", "b", "c", "d", "e"],
        "onnx_sha256": sha256 or hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def test_int8_requires_calibration_dataset(tmp_path):
    config = RKNNBuildConfig(target="rk3588", quantize=8, batch=1)

    with pytest.raises(ValueError, match="calibration"):
        convert_onnx_to_rknn(tmp_path / "model.onnx", tmp_path / "model.rknn", config)


def test_calibration_list_contains_absolute_sorted_images(tmp_path):
    second = tmp_path / "b.jpg"
    first = tmp_path / "a.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    output = write_calibration_list([second, first], tmp_path / "dataset.txt")

    assert output.read_text(encoding="utf-8").splitlines() == [str(first.resolve()), str(second.resolve())]


def test_fp16_conversion_configures_normalization_and_writes_metadata(monkeypatch, tmp_path):
    onnx_path = tmp_path / "raw.onnx"
    onnx_path.write_bytes(b"raw")
    _write_source_metadata(onnx_path)
    monkeypatch.setattr(
        "gauge_detector.rknn_export._onnx_contract", lambda _: ([[1, 3, 544, 960]], [[1, 9, 10710]])
    )
    monkeypatch.setattr("gauge_detector.rknn_export._toolkit_version", lambda: "2.3.2")

    class FakeRKNN:
        def config(self, **kwargs):
            self.config_options = kwargs
            return 0

        def load_onnx(self, **kwargs):
            self.load_options = kwargs
            return 0

        def build(self, **kwargs):
            self.build_options = kwargs
            return 0

        def export_rknn(self, path):
            Path(path).write_bytes(b"rknn")
            return 0

        def release(self):
            self.released = True

    runtime = FakeRKNN()
    output = convert_onnx_to_rknn(
        onnx_path,
        tmp_path / "model.rknn",
        RKNNBuildConfig(),
        rknn_factory=lambda: runtime,
    )

    assert runtime.config_options == {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": "rk3588",
    }
    assert runtime.build_options == {"do_quantization": False, "rknn_batch_size": 1}
    assert runtime.released is True
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["quantize"] == 16
    assert metadata["source_output_shapes"] == [[1, 9, 10710]]
    assert metadata["toolkit_version"] == "2.3.2"
    assert metadata["input_shape"] == [544, 960]
    assert metadata["input_layout"] == "NHWC"
    assert metadata["input_dtype"] == "uint8"
    assert metadata["raw_head"] == "one2one"
    assert metadata["prompts"] == ["a", "b", "c", "d", "e"]


def test_conversion_rejects_end2end_onnx(monkeypatch, tmp_path):
    onnx_path = tmp_path / "end2end.onnx"
    onnx_path.write_bytes(b"end2end")
    _write_source_metadata(onnx_path)
    monkeypatch.setattr(
        "gauge_detector.rknn_export._onnx_contract", lambda _: ([[1, 3, 544, 960]], [[1, 300, 6]])
    )

    with pytest.raises(ValueError, match="end-to-end"):
        convert_onnx_to_rknn(onnx_path, tmp_path / "model.rknn", RKNNBuildConfig(), rknn_factory=lambda: object())


def test_conversion_rejects_untrusted_or_one2many_source_metadata(monkeypatch, tmp_path):
    onnx_path = tmp_path / "raw.onnx"
    onnx_path.write_bytes(b"raw")
    _write_source_metadata(onnx_path, raw_head="one2many")
    monkeypatch.setattr(
        "gauge_detector.rknn_export._onnx_contract", lambda _: ([[1, 3, 544, 960]], [[1, 9, 10710]])
    )

    with pytest.raises(ValueError, match="one2one"):
        convert_onnx_to_rknn(onnx_path, tmp_path / "model.rknn", RKNNBuildConfig(), rknn_factory=lambda: object())


def test_conversion_rejects_dynamic_or_multiple_tensor_contract(monkeypatch, tmp_path):
    onnx_path = tmp_path / "raw.onnx"
    onnx_path.write_bytes(b"raw")
    _write_source_metadata(onnx_path)
    monkeypatch.setattr(
        "gauge_detector.rknn_export._onnx_contract",
        lambda _: ([["batch", 3, 544, 960]], [[1, 9, 10710], [1, 1, 1]]),
    )

    with pytest.raises(ValueError, match="static input"):
        convert_onnx_to_rknn(onnx_path, tmp_path / "model.rknn", RKNNBuildConfig(), rknn_factory=lambda: object())


@pytest.mark.parametrize("bad_operation", ["config", "load_onnx", "build", "export_rknn"])
@pytest.mark.parametrize("bad_code", [None, -1])
def test_conversion_strictly_checks_every_return_code_and_releases(
    monkeypatch, tmp_path, bad_operation, bad_code
):
    onnx_path = tmp_path / "raw.onnx"
    onnx_path.write_bytes(b"raw")
    _write_source_metadata(onnx_path)
    monkeypatch.setattr(
        "gauge_detector.rknn_export._onnx_contract", lambda _: ([[1, 3, 544, 960]], [[1, 9, 10710]])
    )

    class BadRuntime:
        released = False

        def _result(self, operation):
            return bad_code if operation == bad_operation else 0

        def config(self, **kwargs):
            return self._result("config")

        def load_onnx(self, **kwargs):
            return self._result("load_onnx")

        def build(self, **kwargs):
            return self._result("build")

        def export_rknn(self, path):
            Path(path).write_bytes(b"rknn")
            return self._result("export_rknn")

        def release(self):
            self.released = True

    runtime = BadRuntime()
    with pytest.raises(RuntimeError, match=bad_operation):
        convert_onnx_to_rknn(
            onnx_path, tmp_path / "model.rknn", RKNNBuildConfig(), rknn_factory=lambda: runtime
        )

    assert runtime.released is True


def test_setup_script_pins_toolkit_version():
    script = Path("scripts/setup_rknn_env.sh")

    assert script.is_file()
    contents = script.read_text(encoding="utf-8")
    assert "rknn-toolkit2==2.3.2" in contents
    assert '"rknn-toolkit2==2.3.2" --no-deps' in contents
    assert "torch==" not in contents


def test_cli_parser_accepts_rknn_export_commands():
    from gauge_detector.cli import build_parser

    parser = build_parser()
    source_args = parser.parse_args(
        ["export-rknn-onnx", "--config", "config.yaml", "--profile", "prompts.npz", "--output", "artifacts"]
    )
    convert_args = parser.parse_args(
        [
            "convert-rknn",
            "--onnx",
            "raw.onnx",
            "--output",
            "model.rknn",
            "--target",
            "rk3588",
            "--quantize",
            "16",
        ]
    )

    assert source_args.command == "export-rknn-onnx"
    assert convert_args.command == "convert-rknn"
    assert convert_args.quantize == 16
