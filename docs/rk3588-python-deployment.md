# RK3588 Python 部署与板端验收

## 1. 已冻结的模型契约

- RKNN-Toolkit2：2.3.2；板端 RKNN-Toolkit-Lite2、`librknnrt` 和 RKNPU 驱动必须使用板卡厂商验证过的兼容组合。
- 模型：YOLOE-26s、静态 Text Prompt、FP16、batch=1。
- 输入：RGB uint8 NHWC，由应用执行 letterbox；静态 Shape 为高 544、宽 960。
- 归一化：RKNN 模型内置 `mean=[0,0,0]`、`std=[255,255,255]`，应用不得再次除以 255。
- 输出：one-to-one raw `(1,9,10710)`，前 4 通道为 `xywh`，后 5 通道为 Prompt 分数。
- 业务规则：类别统一为 `instrument`，类别无关 NMS，每张图最多一个目标，嵌套组保留最大框。

如采集分辨率或长宽比不再是 1920×1080，应先在主机重新选择并验证静态 `input_shape`，不能仅在板端拉伸到 544×960。

## 2. 主机生成物

在项目根目录执行：

```bash
bash scripts/setup_rknn_env.sh
PYTHONPATH=src .venv-rknn/bin/python -m gauge_detector convert-rknn \
  --onnx artifacts/rk3588/yoloe-26s-rknn-source.onnx \
  --output artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --target rk3588 --quantize 16
```

至少复制以下内容到 RK3588：

```text
artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn
configs/rk3588.yaml
scripts/run_rknn_detection.sh
src/gauge_detector/
pyproject.toml
```

`.rknn` 和 `.onnx` 是本地生成的大文件，默认不提交 Git；相邻 JSON 元数据已提交，用于核对来源、Shape、Prompt 和 SHA256。

## 3. 板端 Python 环境

建议使用板卡系统自带或厂商验证的 Python 3.10/3.11 环境。安装与系统架构、Python ABI 和 Runtime 匹配的 `rknn-toolkit-lite2` wheel，再安装：

```bash
python3 -m pip install "numpy<2" "opencv-python-headless>=4.8" "PyYAML>=6"
python3 -m pip install /path/to/rknn_toolkit_lite2-2.3.2-*.whl
python3 -m pip install -e . --no-deps
```

环境检查：

```bash
python3 - <<'PY'
import cv2, numpy, yaml
from rknnlite.api import RKNNLite
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
print("RKNNLite import OK", RKNNLite)
PY
```

同时记录板端系统镜像、内核、RKNPU 驱动和 `librknnrt` 版本。若 `init_runtime` 报模型版本或驱动不兼容，停止测试并按板卡厂商版本矩阵重新安装 Runtime 或用匹配 Toolkit 重新转换，不自动覆盖系统库。

## 4. 单图与 11 图回归

单图冒烟测试：

```bash
bash scripts/run_rknn_detection.sh \
  --model artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --input /path/to/detect-01.jpg \
  --output outputs/rknn-smoke \
  --core-mask AUTO
```

11 图回归：

```bash
bash scripts/run_rknn_detection.sh \
  --model artifacts/rk3588/yoloe-26s-rk3588-fp16.rknn \
  --input /path/to/检测采图 \
  --output outputs/rknn-11 \
  --core-mask AUTO
```

验收条件：

- 11 张图均只有一个业务检测框，无重复框。
- `detect-01`、`detect-09`、`detect-10` 必须检出。
- 相对 `outputs/rk3588_host_baseline` 的最终框 IoU 均不低于 0.80。
- 人工检查 11 张 annotated 图片，尤其检查 `detect-02` 是否选择右侧完整仪表、`detect-09` 是否保留完整嵌套组。

## 5. 延迟与 core-mask 测试

先用 `AUTO` 完成正确性验收，再依次测试：

```text
AUTO
CORE_0
CORE_1
CORE_2
CORE_0_1
CORE_0_1_2
```

每种模式使用同一批图片，前 3 次推理只用于预热，随后至少运行 50 次。分别记录 letterbox、`RKNNLite.inference()`、Python 解码/业务后处理和端到端耗时，报告中位数、P90 和 P99；图片读取时间单独记录，不混入模型端到端时间。不要根据单次最快值选择 core mask。

板端 Python 可用 `GaugeDetector("configs/rk3588.yaml", warmup_runs=3)` 保持模型只加载一次。`DetectionResult.inference_ms` 只表示 RKNN Runtime 调用耗时；端到端耗时需在 `detect_array()` 外层用 `time.perf_counter()` 测量。

## 6. 30 分钟稳定性测试

使用最终选定的 core mask 连续循环 11 张图片至少 30 分钟，模型实例不得在循环内重建。每分钟记录：

- 已处理帧数、失败数、检测数；
- inference 中位数/P90/P99；
- SoC/NPU 温度和 CPU/NPU 频率；
- 进程 RSS；
- 内核日志中的 RKNPU reset、timeout 或 OOM。

验收时要求无 Runtime 错误、无持续内存增长、关键图片无间歇漏检。若热降频导致延迟上升，应先改善散热和电源，再讨论降低输入尺寸；768/640 只能重新导出并完成全量精度回归后采用。

## 7. 当前尚需板端完成

- 真实 RK3588 NPU 的 11 图精度回归；
- `AUTO` 与各 core mask 的延迟对比；
- Runtime/驱动兼容性确认；
- 30 分钟温度与稳定性测试。

主机 RKNN 模拟器结果只能证明转换图可执行，不能替代以上项目。
