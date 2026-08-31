# YOLOE Text Prompt 工业仪表检测

本项目使用普通 YOLOE `*-seg.pt` checkpoint 和 Text Prompt Ensemble，在不训练、不微调的情况下检测工业仪表。多个开放词汇 Prompt 只用于提高召回率，业务输出统一为 `class_id=0, class_name=instrument`。

不再需要参考图片、人工选择 ROI 或 Visual Prompt Profile。虽然模型是 segmentation checkpoint，业务代码只读取 `result.boxes.xyxy`，不使用 mask。

## 检测流程

```text
启动程序
  → 加载一次 YOLOE
  → set_classes(text_prompts) 一次
  → 输入图片
  → Text Prompt Ensemble 推理
  → class-agnostic NMS
  → 统一映射为 instrument
  → 从左到右排序
  → 不同颜色框、编号、JSON 和 crops
```

默认 Prompt 在 `configs/default.yaml` 中集中维护：

```yaml
text_prompt:
  prompts:
    - analog gauge
    - dial gauge
    - pressure gauge
    - pressure meter
    - industrial gauge
  unified_class_name: instrument
```

## 安装

```bash
cd /home/mcl/code/YOLO/yoloe_gauge_detector
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
```

项目已固定验证 Ultralytics 8.4.121。Text Prompt 第一次初始化可能安装 MobileCLIP 并下载对应 text encoder 权重；初始化失败时程序会明确报错，不会使用未初始化的模型继续推理。

模型必须是普通 `*-seg.pt`，不能使用不支持 `set_classes()` 的 `*-seg-pf.pt` Prompt-Free checkpoint。

## 配置

主要参数位于 `configs/default.yaml`：

```yaml
model:
  name: yoloe-26s-seg.pt
  device: auto
  imgsz: 960
  half: true
detection:
  conf: 0.15
  iou: 0.50
  agnostic_nms: true
  max_det: 20
```

`agnostic_nms` 必须为 `true`，否则同一仪表可能因匹配多个 Prompt 而产生重复框。项目还会在业务输出边界使用同一个 `iou` 阈值做一次类别无关去重，避免不同 Ultralytics 版本或 task predictor 的行为差异泄漏重复框。无 CUDA 时自动回退 CPU，并关闭 FP16。

当前阶段假定每张图片最多只有一个目标仪表。`postprocess.single_target` 会先选择置信度最高的候选；如果它与其他框的较小框覆盖率达到 `containment_threshold`，则改为保留该嵌套组中面积最大的框。将 `enabled` 改为 `false` 可恢复多目标输出。

## 检测单图

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector detect \
  --image data/demo/demo.jpg \
  --output outputs \
  --config configs/default.yaml
```

或直接运行：

```bash
bash scripts/demo.sh
```

## 一键启动

直接运行脚本会使用 `scripts/run_detection.sh` 顶部的默认输入、输出和配置路径：

```bash
bash scripts/run_detection.sh
```

也可以在命令行覆盖输入和输出。脚本会自动判断输入是单张图片还是图片文件夹：

```bash
bash scripts/run_detection.sh \
  --input "/home/mcl/data/yolo/检测采图" \
  --output "outputs/my_results"
```

递归处理子文件夹：

```bash
bash scripts/run_detection.sh \
  --input "/path/to/images" \
  --output "outputs/recursive_results" \
  --recursive
```

查看所有参数：

```bash
bash scripts/run_detection.sh --help
```

## 批量检测

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector detect-dir \
  --input data/test_images \
  --output outputs/test_images \
  --config configs/default.yaml
```

默认不递归；加入 `--recursive` 可递归扫描 `.jpg/.jpeg/.png/.bmp/.webp`。同一个 `detect-dir` 进程只加载一次模型、初始化一次 Prompt。

## Python API

```python
from gauge_detector import GaugeDetector

detector = GaugeDetector("configs/default.yaml")
result = detector.detect("data/demo/demo.jpg")
result_from_camera = detector.detect_array(bgr_frame)
for item in result.detections:
    print(item.class_name, item.confidence, item.xyxy)
```

`detect_array()` 接收 OpenCV BGR `ndarray`，可直接用于摄像头或现有 API。模型实例封装在 `GaugeDetector` 中，不会每帧重新初始化。

## 输出

```text
outputs/
├── demo_annotated.jpg
├── demo.json
└── crops/
    ├── demo_instrument_00.jpg
    ├── demo_instrument_01.jpg
    └── demo_instrument_02.jpg
```

JSON 保持原有坐标结构，类别统一为 `instrument`：

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "instrument",
      "confidence": 0.91,
      "bbox_xyxy": [100, 100, 300, 300]
    }
  ]
}
```

没有检测结果时 `detections` 为 `[]`。可视化框使用不同颜色，并按检测框中心从左到右编号。

## Benchmark

```bash
bash scripts/benchmark.sh
```

Benchmark 复用同一个 Text Prompt Detector，扫描 `configs/benchmark.yaml` 中的 `imgsz` 和 `conf`，报告 TP、FP、FN、Precision、Recall、F1 和延迟。

## 导出

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector export \
  --format onnx \
  --config configs/default.yaml
```

导出前会加载模型并初始化相同的 Text Prompt。TensorRT 需要 NVIDIA GPU、匹配驱动和相关依赖。

## RK3588 Python 部署

默认 `configs/default.yaml` 始终使用稳定的 PyTorch 后端。RK3588 部署使用独立的 `configs/rk3588.yaml`，静态输入为 `1×3×544×960`；该尺寸与当前 1920×1080 采图在 `imgsz=960` 下的最小矩形预处理一致，不能改成 960×960 后直接沿用当前验收结果。

主机准备流程：

```bash
PYTHONPATH=src .venv/bin/python -m gauge_detector prepare-profile \
  --config configs/default.yaml \
  --output artifacts/rk3588/gauge-prompts.npz

PYTHONPATH=src .venv/bin/python -m gauge_detector export-rknn-onnx \
  --config configs/onnx.yaml \
  --profile artifacts/rk3588/gauge-prompts.npz \
  --output artifacts/rk3588

bash scripts/setup_rknn_env.sh
PYTHONPATH=src .venv-rknn/bin/python -m gauge_detector convert-rknn \
  --onnx artifacts/rk3588/yoloe-26s-rknn-source.onnx \
  --output artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --target rk3588 --quantize 16
```

RKNN 源 ONNX 使用 YOLO26 one-to-one 检测头的 raw 输出 `(1,9,10710)`，只把 RKNN 不支持的 TopK/NMS 移到 Python；不要使用主机校验模型的 `(1,300,6)` 输出，也不要切换到 one-to-many 头。

板端运行：

```bash
bash scripts/run_rknn_detection.sh \
  --model artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --input "/path/to/images" \
  --output outputs/rknn \
  --core-mask AUTO
```

完整的环境匹配、复制清单、逐核测试和稳定性验收见 `docs/rk3588-python-deployment.md`。主机测试证据见 `artifacts/rk3588/host-validation.md`。

## 测试

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

实际模型集成测试默认跳过；设置 `YOLOE_RUN_INTEGRATION=1` 后运行。首次 Text Prompt 初始化需要 text encoder 资源可用。
