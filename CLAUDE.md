# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

使用 YOLOE Text Prompt 做零样本工业仪表检测：不训练、不微调、不使用参考图或 Visual Prompt。虽然加载的是 `*-seg.pt` 分割 checkpoint，业务代码只读取 `result.boxes.xyxy`，不使用 mask。多个开放词汇 Prompt 仅用于提高召回率，所有检测结果统一为 `class_id=0, class_name=instrument`。

板端相关的文档和用户可见文案（`board_cli.py`、`run_rknn_detection.sh`、`docs/`）使用中文；库代码、日志和主机端 CLI 文案使用英文。修改时与所在文件保持一致。

## 常用命令

```bash
# 测试（无需模型和网络，约 1 秒）
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/test_backends.py::test_name

# 需要真实 checkpoint 的集成测试（默认跳过）
YOLOE_RUN_INTEGRATION=1 PYTHONPATH=src .venv/bin/python -m pytest -m integration

# 检测
PYTHONPATH=src .venv/bin/python -m gauge_detector detect --image X.jpg --output outputs --config configs/default.yaml
PYTHONPATH=src .venv/bin/python -m gauge_detector detect-dir --input DIR --output outputs [--recursive]
bash scripts/run_detection.sh [--input PATH] [--output PATH] [--recursive]   # 自动判断输入是单图还是目录
bash scripts/benchmark.sh
```

两个虚拟环境必须保持隔离：`.venv`（Python 3.11、Ultralytics/PyTorch、`requirements.lock`）用于所有主机端工作；`.venv-rknn`（Python 3.10、`rknn-toolkit2==2.3.2`，由 `scripts/setup_rknn_env.sh` 创建）**只**用于 `convert-rknn`。不要把 rknn-toolkit2 装进 `.venv`。

Ultralytics 固定为 `>=8.4.120,<8.5`，已验证版本是 8.4.121。

## 架构

### 运行时流程

`GaugeDetector`（`detector.py`）在整个生命周期内持有配置和唯一一个 backend 实例，模型不会每帧重新初始化。`detect_array()` 接收 OpenCV BGR ndarray，是 `detect()` 以及摄像头/API 调用方的共同入口。

Backend（`backends.py`，由 YAML 中的 `model.backend` 选择）统一返回 `BackendPrediction(detections, inference_ms)`：

- `pytorch` —— 经 `model.py` 调用 Ultralytics YOLOE，构造时执行一次 `set_classes()`。只有这条路径需要 torch。
- `onnx` —— onnxruntime，要求静态 `1x3xHxW` 输入和 `1xNx6` 端到端输出。
- `rknn` —— 板端 RKNNLite，单个 raw 输出 `1x(4+len(prompts))xAnchors`，TopK/NMS 在 Python 侧完成。

每个 backend 都接受可注入的工厂参数（`model_factory` / `session_factory` / `runtime_factory`），测试套件正是靠它在没有模型文件、运行时和硬件的情况下覆盖全部三条路径。修改时务必保留这个接缝。

`detector.py` 中的后处理顺序固定且与 backend 无关：`remove_duplicate_boxes`（在业务边界再做一次类别无关 IoU 去重，避免不同版本 predictor 行为泄漏重复框）→ `filter_geometry` → `select_single_target` → `sort_detections`（从左到右）。

`detection.agnostic_nms` 必须为 `true`，否则 detector 直接报错：同一仪表匹配多个 Prompt 会产生重复框。

`postprocess.single_target` 固化了当前业务约束「每张图最多一个仪表」：先取置信度最高的框，再扩展为其嵌套组（覆盖率 ≥ 阈值）中面积最大的框。设为 `enabled: false` 可恢复多目标输出。

### 配置

`configs/default.yaml`（pytorch，主机）、`configs/onnx.yaml`、`configs/rk3588.yaml` 都会合并到 `config.py` 的 `DEFAULT_CONFIG` 之上，因此 YAML 只需写出差异项。三份配置中的 prompts 是重复的，必须保持一致：`rknn_export.py` 和 `build_rk3588_package.py` 会把配置里的 prompt 列表与固化进模型的元数据做比对，不一致直接失败。

`input_shape: [544, 960]` 不是随意取值，而是 1920×1080 采图在 `imgsz=960` 下的最小矩形。采集分辨率或长宽比变化时，必须先在主机重新选择并验证静态 shape；改成 960×960 会使现有验收结果失效。

### 主机到板端的导出链

```
prepare-profile   → .npz Prompt 嵌入 + .json 元数据（checkpoint sha256、prompts、imgsz）
export-rknn-onnx  → 静态 raw 输出 ONNX + .json（追加 onnx sha256、input_shape、raw_head）
convert-rknn      → .rknn + .json（追加 toolkit 版本、归一化参数、rknn sha256）   [.venv-rknn]
build_rk3588_package.py → dist/ 压缩包 + SHA256SUMS
```

每一步都会先校验上一步的 JSON 才继续，因此整条链只能端到端复现。`.npz`/`.onnx`/`.rknn` 被 Git 忽略；同名 JSON 已提交，是唯一的来源与契约记录。

代码中已强制、不得漂移的契约：

- RKNN 源 ONNX 使用 YOLO26 **one-to-one** 头的 raw 输出（`export.py` 的 `use_one2one_raw_output` 把 `cv2/cv3/cv4` 换成对应的 `one2one_*` 并关闭 `end2end`）。不要把主机校验用的 `1xNx6` 模型送进 `convert-rknn`，也不要切换到 one-to-many 头 —— `rknn_export.py` 会拒绝这两种情况。
- 归一化内置在 RKNN 模型中（`mean=[0,0,0]`、`std=[255,255,255]`），板端输入 uint8 NHWC RGB，**不能**再除以 255。RKNN 走 `letterbox_rgb`（uint8），ONNX 走 `onnx_tensor`（float，/255）。
- Prompt 已固化进模型。修改 Prompt、权重或输入 shape 必须重跑整条导出链；只改 YAML 会得到与模型不匹配的配置。

### 板端运行时子集

`scripts/build_rk3588_package.py` 中的 `RUNTIME_MODULES` 列出了唯一会被复制到 RK3588 的模块：不含 `model.py`、`export.py`、`cli.py`、`benchmark.py`、`compare.py`。因此这些被打包的模块不能在模块级 import torch、ultralytics 或 onnx —— `backends.py` 之所以把 `.model` 延迟到 `PyTorchBackend.__init__` 内部导入，正是为此。板端入口是 `gauge_detector.board_cli`（配置默认 `configs/rk3588.yaml`，以上下文管理器方式使用 `GaugeDetector`，确保 `RKNNLite.release()` 一定执行）。

`compare-backends` 在图片目录上比对两份配置（逐图框数 + IoU），是新 backend 对齐 PyTorch 基线的验收方式。

## 测试

`tests/` 与模块一一对应，且无需 checkpoint 即可运行。Shell 脚本的测试方式是把脚本复制进伪造的项目目录树，用记录 argv 的假 `python` 替身来验证参数（见 `tests/test_run_script.py`）—— 扩展时沿用这个模式，不要真正执行推理流程。集成测试同时带 `@pytest.mark.integration` 和 `YOLOE_RUN_INTEGRATION` 的 skipif。

## 参考文档

- `docs/rk3588-python-deployment.md` —— 完整的板端环境、逐核测试和稳定性验收流程。
- `docs/RK3588_MIGRATION_GUIDE.md` —— 会以 `MIGRATION_GUIDE.md` 的名字打进迁移包。
- `artifacts/rk3588/host-validation.md` —— 主机端测试证据。
