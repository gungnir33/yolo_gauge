# RK3588 Python Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持现有 PyTorch 仪表检测结果和业务输出稳定的同时，增加静态 Prompt、纯检测 ONNX、RKNN 转换和 RKNNLite Python 推理链路。

**Architecture:** 现有 `GaugeDetector` 继续负责统一后处理和输出；PyTorch、ONNX Runtime、RKNNLite 通过统一后端接口提供候选框。Prompt 编码和模型转换只在主机离线执行，RK3588 运行时仅加载静态 `.rknn` 模型。

**Tech Stack:** Python 3.13（现有 PyTorch/ONNX 环境）、Python 3.10（RKNN-Toolkit2 环境）、Ultralytics 8.4.121、ONNX、ONNX Runtime、RKNN-Toolkit2 2.3.2、RKNN-Toolkit-Lite2、OpenCV、NumPy、pytest。

**Spec:** `docs/superpowers/specs/2026-08-31-rk3588-python-migration-design.md`

## Global Constraints

- 不改变 `configs/default.yaml` 的默认 PyTorch 行为。
- 不改变现有单目标、嵌套框、JSON、裁剪和绘图语义。
- 每个生产代码变化必须先有失败测试，并观察到预期失败。
- 每个任务完成后运行该任务的定向测试和全量单元测试。
- 第一版固定 `batch=1`、PyTorch/Profile `imgsz=960`、部署静态 Shape `1×3×544×960`、RK3588、FP16。
- `detect-01`、`detect-09`、`detect-10` 必须检出，最终框相对主机基线 IoU 不低于 0.80。
- 当前 `.venv` 不安装 RKNN-Toolkit2；RKNN 转换使用独立 Python 3.10 环境。
- RK3588 真实推理、NPU 延迟、多核和温度测试必须在板端完成，不以主机模拟结果替代。

---

## File Structure

- Create `src/gauge_detector/prompt_profile.py`: Prompt profile 元数据、SHA256 和保存校验。
- Create `src/gauge_detector/preprocess.py`: ONNX/RKNN 共用 letterbox 和坐标恢复。
- Create `src/gauge_detector/runtime_output.py`: YOLOE-26 end-to-end 输出解析。
- Create `src/gauge_detector/backends.py`: PyTorch、ONNX Runtime、RKNNLite 后端及工厂。
- Create `src/gauge_detector/rknn_export.py`: ONNX 到 RKNN 的转换参数和可选依赖边界。
- Modify `src/gauge_detector/model.py`: 增加保存/加载 Prompt Embedding 的薄封装。
- Modify `src/gauge_detector/export.py`: 增加纯检测架构映射、静态 ONNX 和 RKNN 导出入口。
- Modify `src/gauge_detector/config.py`: 增加后端和 RKNN 默认配置并验证关键字段。
- Modify `src/gauge_detector/detector.py`: 从后端工厂取候选框，复用现有后处理。
- Modify `src/gauge_detector/cli.py`: 增加 `prepare-profile`、`export-onnx`、`convert-rknn`、`compare-backends`。
- Create `configs/rk3588.yaml`: 板端 Python 配置。
- Create `scripts/setup_rknn_env.sh`: 独立 Python 3.10 转换环境安装脚本。
- Create `scripts/run_rknn_detection.sh`: RK3588 Python 一键运行脚本。
- Create `tests/test_prompt_profile.py`: profile 行为测试。
- Create `tests/test_preprocess.py`: letterbox 和坐标恢复测试。
- Create `tests/test_runtime_output.py`: end-to-end 输出解析测试。
- Create `tests/test_backends.py`: ONNX/RKNNLite 生命周期和错误行为测试。
- Create `tests/test_rknn_export.py`: 转换配置、校准清单和缺依赖测试。
- Create `tests/test_export.py`: 纯检测架构映射和导出调用测试。
- Create `tests/test_backend_comparison.py`: IoU 和比较报告测试。
- Modify `README.md`: 主机准备、转换、板端运行和验收命令。

---

### Task 1: Prompt Profile Artifact

**Files:**
- Create: `src/gauge_detector/prompt_profile.py`
- Modify: `src/gauge_detector/model.py`
- Create: `tests/test_prompt_profile.py`

**Interfaces:**
- Produces: `checkpoint_sha256(path: str | Path) -> str`
- Produces: `PromptProfileMetadata.create(checkpoint, prompts, imgsz) -> PromptProfileMetadata`
- Produces: `save_profile_metadata(path, metadata) -> Path`
- Produces: `load_profile_metadata(path) -> PromptProfileMetadata`
- Produces: `validate_profile(metadata, checkpoint, prompts, imgsz) -> None`
- Produces: `YOLOEModel.save_prompt_embeddings(path) -> Path`
- Produces: `YOLOEModel.load_prompt_embeddings(path, prompts) -> None`

- [ ] **Step 1: Write failing metadata tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify expected failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompt_profile.py -v`

Expected: collection fails because `gauge_detector.prompt_profile` does not exist.

- [ ] **Step 3: Implement immutable profile metadata and validation**

Use a frozen dataclass with schema version `1`, resolved checkpoint name, checkpoint SHA256, ordered prompts, and `imgsz`. JSON output must use sorted keys and UTF-8.

- [ ] **Step 4: Run profile tests and full unit tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompt_profile.py -v`

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Expected: all tests pass; integration model test remains skipped unless explicitly enabled.

- [ ] **Step 5: Generate and reload the real profile**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector prepare-profile \
  --config configs/default.yaml \
  --output artifacts/rk3588/gauge-prompts.npz
```

Expected: NPZ and adjacent JSON metadata are created; a new `GaugeDetector` can load the same profile and detect `data/demo/demo.jpg`.

- [ ] **Step 6: Commit**

```bash
git add src/gauge_detector/prompt_profile.py src/gauge_detector/model.py tests/test_prompt_profile.py src/gauge_detector/cli.py
git commit -m "feat: add reproducible YOLOE prompt profiles"
```

---

### Task 2: Detection-Only Static ONNX Export

**Files:**
- Modify: `src/gauge_detector/export.py`
- Modify: `src/gauge_detector/cli.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Consumes: Task 1 profile metadata and NPZ.
- Produces: `detection_yaml_for_checkpoint(checkpoint: str | Path) -> str`
- Produces: `build_detection_only_model(checkpoint, profile_path, prompts) -> Any`
- Produces: `export_detection_onnx(config_path, profile_path, output_dir) -> Path`

- [ ] **Step 1: Write failing architecture mapping tests**

```python
@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [("yoloe-26n-seg.pt", "yoloe-26n.yaml"), ("yoloe-26s-seg.pt", "yoloe-26s.yaml")],
)
def test_detection_yaml_matches_checkpoint_scale(checkpoint, expected):
    assert detection_yaml_for_checkpoint(checkpoint) == expected

def test_detection_yaml_rejects_prompt_free_checkpoint():
    with pytest.raises(ValueError, match="prompt-free"):
        detection_yaml_for_checkpoint("yoloe-26s-seg-pf.pt")
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_export.py -v`

Expected: import fails because `detection_yaml_for_checkpoint` is missing.

- [ ] **Step 3: Implement minimal pure detection exporter**

Build `YOLOE("yoloe-26s.yaml").load("yoloe-26s-seg.pt")`, load the validated NPZ profile, and export with `format="onnx"`, `imgsz=(544,960)`, `batch=1`, `dynamic=False`, `opset=19`, `simplify=False`, `nms=False` and CPU device. Move the resulting file into the requested output directory and save model metadata beside it.

- [ ] **Step 4: Run unit tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_export.py -v`

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

- [ ] **Step 5: Install ONNX host dependencies without changing RKNN environment**

Run:

```bash
.venv/bin/pip install "onnx>=1.16.1,<1.19.0" "onnxruntime>=1.20,<2"
```

Record exact resolved versions in `artifacts/rk3588/host-environment.txt`.

- [ ] **Step 6: Export and inspect the real ONNX model**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector export-onnx \
  --config configs/default.yaml \
  --profile artifacts/rk3588/gauge-prompts.npz \
  --output artifacts/rk3588
```

Run ONNX checker and an ONNX Runtime zero-input inference. Expected input is static `1×3×544×960`; expected end-to-end output is `1×N×6` with `[x1,y1,x2,y2,confidence,class_id]`. If the output contract differs, stop this task and update the design before writing a decoder.

- [ ] **Step 7: Commit**

```bash
git add src/gauge_detector/export.py src/gauge_detector/cli.py tests/test_export.py README.md
git commit -m "feat: export static detection-only YOLOE ONNX"
```

---

### Task 3: Shared Preprocessing and End-to-End Output Decoder

**Files:**
- Create: `src/gauge_detector/preprocess.py`
- Create: `src/gauge_detector/runtime_output.py`
- Create: `tests/test_preprocess.py`
- Create: `tests/test_runtime_output.py`

**Interfaces:**
- Produces: `LetterboxTransform(ratio: float, pad_x: float, pad_y: float, input_size: int)`
- Produces: `letterbox_rgb(image, size, pad_color) -> tuple[np.ndarray, LetterboxTransform]`
- Produces: `onnx_tensor(image, size, pad_color) -> tuple[np.ndarray, LetterboxTransform]`
- Produces: `restore_xyxy(box, transform, image_shape) -> tuple[float, float, float, float]`
- Produces: `decode_end2end_output(output, transform, image_shape, conf) -> list[Detection]`
- Produces: `decode_raw_output(output, transform, image_shape, conf, iou, max_det) -> list[Detection]`

- [ ] **Step 1: Write failing letterbox tests with hand-derived values**

```python
def test_letterbox_records_center_padding_for_wide_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb, transform = letterbox_rgb(image, 200, (114, 114, 114))
    assert rgb.shape == (200, 200, 3)
    assert transform.ratio == 1.0
    assert transform.pad_x == 0.0
    assert transform.pad_y == 50.0

def test_restore_xyxy_reverses_padding_and_scale():
    transform = LetterboxTransform(ratio=0.5, pad_x=10.0, pad_y=20.0, input_size=200)
    assert restore_xyxy([20, 30, 70, 80], transform, (200, 300)) == (20, 20, 120, 120)
```

- [ ] **Step 2: Verify RED for preprocessing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_preprocess.py -v`

- [ ] **Step 3: Implement letterbox and ONNX tensor conversion**

`letterbox_rgb` converts BGR to RGB, resizes with `cv2.INTER_LINEAR`, applies symmetric integer padding and returns contiguous uint8 NHWC. `onnx_tensor` converts this result to contiguous float32 NCHW batch tensor divided by 255 exactly once.

- [ ] **Step 4: Write failing output decoder tests**

```python
def test_decoder_filters_confidence_and_restores_box():
    output = np.array([[[20, 30, 70, 80, 0.9, 3], [0, 0, 5, 5, 0.1, 0]]], dtype=np.float32)
    transform = LetterboxTransform(0.5, 10.0, 20.0, 200)
    detections = decode_end2end_output(output, transform, (200, 300), conf=0.15)
    assert len(detections) == 1
    assert detections[0].xyxy == pytest.approx((20, 20, 120, 120))
    assert detections[0].class_name == "instrument"

def test_decoder_rejects_unknown_output_shape():
    with pytest.raises(ValueError, match="1xNx6"):
        decode_end2end_output(np.zeros((1, 6, 10), dtype=np.float32), transform, (100, 100), 0.15)
```

- [ ] **Step 5: Verify RED, implement decoder, and run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_runtime_output.py -v`

Then implement strict `(1,N,6)` parsing, finite-value checks, thresholding, class unification, coordinate clipping and zero-area rejection.

Add a second failing fixture for RKNN raw output shaped `(1, 4+nc, anchors)`. Implement `xywh→xyxy`, maximum class score selection, class-agnostic NMS and `max_det`; do not implement DFL because YOLOE export has already decoded the box channels before this output boundary.

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_preprocess.py tests/test_runtime_output.py -v`

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add src/gauge_detector/preprocess.py src/gauge_detector/runtime_output.py tests/test_preprocess.py tests/test_runtime_output.py
git commit -m "feat: add portable YOLOE preprocessing and decoding"
```

---

### Task 4: ONNX Runtime Backend and Detector Integration

**Files:**
- Create: `src/gauge_detector/backends.py`
- Modify: `src/gauge_detector/config.py`
- Modify: `src/gauge_detector/detector.py`
- Create: `tests/test_backends.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Produces: `BackendPrediction(detections: list[Detection], inference_ms: float)`
- Produces: `PyTorchBackend.predict(image, conf, iou, max_det) -> BackendPrediction`
- Produces: `ONNXBackend.predict(image, conf, iou, max_det) -> BackendPrediction`
- Produces: `create_backend(config) -> PyTorchBackend | ONNXBackend | RKNNLiteBackend`

- [ ] **Step 1: Write failing backend selection tests**

```python
def test_backend_factory_defaults_to_pytorch():
    backend = create_backend({"model": {"backend": "pytorch", "name": "model.pt", "imgsz": 960, "device": "cpu", "half": False}, "text_prompt": {"prompts": ["dial gauge"]}})
    assert isinstance(backend, PyTorchBackend)

def test_backend_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        create_backend({"model": {"backend": "cuda-magic"}})
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backends.py -v`

- [ ] **Step 3: Implement backend objects and preserve PyTorch behavior**

The PyTorch backend wraps the existing `YOLOEModel` and converts Ultralytics boxes to `Detection`. The ONNX backend creates one `onnxruntime.InferenceSession`, validates static input/output contracts at initialization, uses Task 3 preprocessing/decoder, and reports only session execution time as `inference_ms`.

- [ ] **Step 4: Refactor `GaugeDetector` to consume `BackendPrediction`**

Keep image validation, duplicate removal, geometry filtering, single-target selection and sorting in `GaugeDetector`. Do not duplicate these rules inside a backend.

- [ ] **Step 5: Run focused and full regression tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backends.py tests/test_smoke.py tests/test_postprocess.py -v`

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

- [ ] **Step 6: Run real PyTorch regression on 11 images**

Run:

```bash
bash scripts/run_detection.sh \
  --input "/home/mcl/data/yolo/检测采图" \
  --output "outputs/rk3588_pytorch_after_backend_refactor"
```

Expected: 11/11 images have one detection and all boxes have IoU at least 0.99 against `outputs/rk3588_host_baseline` because the PyTorch path is behavior-preserving. The ONNX static Shape must also reproduce all 11 selected boxes; a square 960×960 input is rejected because it changes candidate ranking on `detect-02`.

- [ ] **Step 7: Commit**

```bash
git add src/gauge_detector/backends.py src/gauge_detector/config.py src/gauge_detector/detector.py tests/test_backends.py tests/test_smoke.py
git commit -m "feat: add selectable PyTorch and ONNX backends"
```

---

### Task 5: Backend Comparison Report

**Files:**
- Create: `src/gauge_detector/compare.py`
- Modify: `src/gauge_detector/cli.py`
- Create: `tests/test_backend_comparison.py`

**Interfaces:**
- Produces: `compare_results(reference, candidate) -> dict[str, Any]`
- Produces: `compare_directory(reference_config, candidate_config, image_dir, output_dir) -> Path`

- [ ] **Step 1: Write failing comparison tests**

```python
def test_compare_results_reports_hand_checked_iou():
    reference = DetectionResult("a.jpg", 100, 100, [Detection(0, "instrument", 0.9, 0, 0, 10, 10)], 10)
    candidate = DetectionResult("a.jpg", 100, 100, [Detection(0, "instrument", 0.8, 5, 0, 15, 10)], 5)
    row = compare_results(reference, candidate)
    assert row["iou"] == pytest.approx(1 / 3)
    assert row["reference_count"] == 1
    assert row["candidate_count"] == 1
```

- [ ] **Step 2: Verify RED, implement report, and run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backend_comparison.py -v`

Report JSON and Markdown with per-image counts, boxes, confidence, IoU, inference time and pass/fail. The summary must separately list missed images and images below the IoU threshold.

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

- [ ] **Step 3: Run real ONNX comparison**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector compare-backends \
  --reference configs/default.yaml \
  --candidate configs/onnx.yaml \
  --images "/home/mcl/data/yolo/检测采图" \
  --output outputs/rk3588_onnx_comparison
```

Expected: 11/11 detected; `detect-01/09/10` detected; all selected boxes IoU at least 0.80. Stop before RKNN work if this gate fails.

- [ ] **Step 4: Commit**

```bash
git add src/gauge_detector/compare.py src/gauge_detector/cli.py tests/test_backend_comparison.py
git commit -m "feat: compare deployment backends against baseline"
```

---

### Task 6: RKNN Conversion Tooling

**Files:**
- Create: `src/gauge_detector/rknn_export.py`
- Modify: `src/gauge_detector/cli.py`
- Create: `configs/rk3588.yaml`
- Create: `scripts/setup_rknn_env.sh`
- Create: `tests/test_rknn_export.py`

**Interfaces:**
- Produces: `RKNNBuildConfig(target="rk3588", quantize=16, batch=1)`
- Produces: `write_calibration_list(images, output) -> Path`
- Produces: `convert_onnx_to_rknn(onnx_path, output_path, config, dataset=None) -> Path`

- [ ] **Step 1: Write failing conversion boundary tests**

```python
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
    assert output.read_text().splitlines() == [str(first.resolve()), str(second.resolve())]
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rknn_export.py -v`

- [ ] **Step 3: Implement optional RKNN dependency boundary**

Import `rknn.api.RKNN` only inside `convert_onnx_to_rknn`. The source ONNX must be separately exported from the one-to-one head with TopK disabled; reject `1×N×6` end-to-end models before conversion and do not fall back to the one-to-many head. Configure `mean_values=[[0,0,0]]`, `std_values=[[255,255,255]]`, `target_platform="rk3588"`; call `load_onnx`, `build(do_quantization=False, rknn_batch_size=1)` for FP16, check every return code, export the model, release in `finally`, and save build metadata JSON.

- [ ] **Step 4: Run tests in the existing environment**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rknn_export.py -v`

Expected: configuration and dataset tests pass; a conversion attempt reports a clear missing `rknn-toolkit2` message without breaking imports elsewhere.

- [ ] **Step 5: Create isolated Python 3.10 conversion environment**

Run: `bash scripts/setup_rknn_env.sh`

The script creates `.venv-rknn`, installs exactly `rknn-toolkit2==2.3.2`, `onnx>=1.16.1,<1.19.0`, `setuptools<82`, NumPy and PyYAML, and prints installed versions.

- [ ] **Step 6: Convert the real FP16 model on the host**

Run:

```bash
PYTHONPATH=src .venv-rknn/bin/python -m gauge_detector convert-rknn \
  --onnx artifacts/rk3588/yoloe-26s-rknn-source.onnx \
  --output artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --target rk3588 \
  --quantize 16
```

Expected: `.rknn` and build metadata are created. Run RKNN simulator inference if supported by the installed toolkit and record it explicitly as simulator output.

- [ ] **Step 7: Run full tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

```bash
git add src/gauge_detector/rknn_export.py src/gauge_detector/cli.py configs/rk3588.yaml scripts/setup_rknn_env.sh tests/test_rknn_export.py
git commit -m "feat: add reproducible RK3588 model conversion"
```

---

### Task 7: RKNNLite Python Backend

**Files:**
- Modify: `src/gauge_detector/backends.py`
- Modify: `src/gauge_detector/config.py`
- Create: `scripts/run_rknn_detection.sh`
- Modify: `tests/test_backends.py`

**Interfaces:**
- Produces: `RKNNLiteBackend(model_path, imgsz, pad_color, core_mask="AUTO", runtime_factory=None)`
- Produces: `RKNNLiteBackend.predict(image, conf, iou, max_det) -> BackendPrediction`
- Produces: `RKNNLiteBackend.close() -> None`

- [ ] **Step 1: Write failing lifecycle test with a complete fake Runtime contract**

```python
def test_rknnlite_backend_initializes_once_and_releases():
    runtime = FakeRKNNLite(
        load_result=0,
        init_result=0,
        outputs=[np.array([[[10, 20, 30, 40, 0.9, 0]]], dtype=np.float32)],
    )
    backend = RKNNLiteBackend("model.rknn", 100, (114, 114, 114), runtime_factory=lambda: runtime)
    prediction = backend.predict(np.zeros((100, 100, 3), dtype=np.uint8), conf=0.15, iou=0.5, max_det=20)
    backend.close()
    assert len(prediction.detections) == 1
    assert runtime.load_calls == ["model.rknn"]
    assert runtime.init_calls == ["AUTO"]
    assert runtime.release_calls == 1
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backends.py -v`

- [ ] **Step 3: Implement RKNNLite backend**

Import `rknnlite.api.RKNNLite` only when the backend is selected. Load and initialize once, use RGB uint8 NHWC input because normalization is baked into RKNN conversion, call `inference(inputs=[input_array])`, decode `1×(4+nc)×anchors` 原始输出并执行类别无关 NMS，and make `close()` idempotent. Map `AUTO`, `CORE_0`, `CORE_1`, `CORE_2`, `CORE_0_1`, `CORE_0_1_2` to RKNNLite constants.

- [ ] **Step 4: Run backend and full tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_backends.py -v`

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

- [ ] **Step 5: Verify board launch script without a board**

Run: `bash scripts/run_rknn_detection.sh --help`

Expected: exits successfully and documents model, input, output, config and core-mask arguments. Running without `rknnlite` must fail with a concise installation message.

- [ ] **Step 6: Commit**

```bash
git add src/gauge_detector/backends.py src/gauge_detector/config.py scripts/run_rknn_detection.sh tests/test_backends.py
git commit -m "feat: add RKNNLite Python inference backend"
```

---

### Task 8: Host Regression, Documentation, and Board Handoff

**Files:**
- Modify: `README.md`
- Create: `docs/rk3588-python-deployment.md`
- Create: `artifacts/rk3588/host-validation.md`

**Interfaces:**
- Consumes all earlier CLI and configuration contracts.
- Produces a complete host validation report and exact RK3588 test commands.

- [ ] **Step 1: Run the complete unit suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Expected: all unit tests pass; only hardware-gated integration tests skip.

- [ ] **Step 2: Run current PyTorch integration suite**

Run: `YOLOE_RUN_INTEGRATION=1 PYTHONPATH=src .venv/bin/python -m pytest -m integration -v`

- [ ] **Step 3: Run all 11 images through PyTorch and ONNX**

Generate separate result directories and a comparison report. Record per-image preprocessing, inference, postprocessing and end-to-end timing after at least 3 warmup runs.

- [ ] **Step 4: Validate critical images**

Assert `detect-01/09/10` each contain exactly one detection and IoU is at least 0.80 relative to `outputs/rk3588_host_baseline`. Inspect the three annotated images visually.

- [ ] **Step 5: Record host-only RKNN status honestly**

Record converter version, generated model SHA256, simulator availability and any unsupported operation warnings. Mark real NPU inference, latency, core-mask comparison and thermal soak as `BOARD_REQUIRED`, not as passed.

- [ ] **Step 6: Write board deployment guide**

Document Python/RKNNLite versions, copy list, environment verification, one-image smoke test, 11-image regression, timing methodology, core-mask matrix and 30-minute stability test.

- [ ] **Step 7: Run final verification and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Run: `git diff --check`

```bash
git add README.md docs/rk3588-python-deployment.md artifacts/rk3588/host-validation.md
git commit -m "docs: add RK3588 Python deployment handoff"
```

- [ ] **Step 8: Request code review and push the feature branch**

Use `superpowers:requesting-code-review`, address findings, rerun final verification, and push the feature branch to `origin` without merging it into `main` until the user reviews host results.
