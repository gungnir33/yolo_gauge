# YOLOE 工业仪表检测 RK3588 Python 迁移说明

## 1. 迁移包说明

本包只包含 RK3588 推理必需的 RKNN 模型、Python 运行时代码、配置和启动脚本，不包含虚拟环境、Python wheel、系统动态库、测试图片、输出图片、PyTorch/ONNX 模型或模型转换工具。

正式迁移交付物是 `yolo_gauge_rk3588_python.tar.gz`。同名展开目录仅用于生成主机上的内容核对，受 Git 忽略，不应单独提交或脱离压缩包复制。模型旁的 JSON 记录模型哈希和部署契约，`SHA256SUMS` 用于检查包内所有文件。

模型契约如下：

- 目标芯片：RK3588；板端 Python：3.8；模型格式：RKNN FP16；batch=1。
- 转换工具：RKNN-Toolkit2 2.3.2。
- 输入：RGB、uint8、NHWC，静态高 544、宽 960。
- 模型内部完成 `mean=[0,0,0]`、`std=[255,255,255]`，应用不能再次除以 255。
- Prompt 已固化到模型中，板端不安装 PyTorch、Ultralytics 或文本编码器。
- 输出继续执行类别无关 NMS、重复框抑制和单目标选择，每张图片最终最多保留一个仪表框。

修改 Prompt、权重或输入 Shape 后必须在主机重新导出 RKNN 模型，不能只编辑 YAML。

## 2. 板端前置条件

当前迁移包已按板端 Python 3.8 适配并锁定兼容依赖。使用板卡厂商验证过的 64 位 Linux 镜像，首先检查：

```bash
uname -m
python3 --version
uname -a
```

`uname -m` 必须为 `aarch64`。记录系统镜像版本、内核、RKNPU 驱动和 `librknnrt` 版本。不同镜像的查询位置可能不同，可依次尝试：

`python3 --version` 应输出 Python 3.8.x。若板端命令不是 `python3`，后续把命令替换成实际的 Python 3.8 解释器路径。

```bash
cat /etc/os-release
sudo dmesg | grep -i rknpu
find /usr -name 'librknnrt.so*' 2>/dev/null
```

本模型由 Toolkit2 2.3.2 生成。板端 RKNN-Toolkit-Lite2、`librknnrt` 和 RKNPU 驱动必须采用板卡厂商确认兼容的组合。不要直接覆盖系统的 `librknnrt.so`；若厂商镜像锁定了其他 Runtime，应优先使用厂商版本矩阵，并在必要时用匹配版本的 Toolkit2 重新生成模型。

## 3. 复制和校验迁移包

把整个压缩包复制到板端，例如：

```bash
scp yolo_gauge_rk3588_python.tar.gz user@RK3588_IP:/opt/
```

在板端解压并校验：

```bash
cd /opt
tar -xzf yolo_gauge_rk3588_python.tar.gz
cd yolo_gauge_rk3588_python
sha256sum -c SHA256SUMS
```

所有文件都应显示 `OK`。校验失败时重新传输，不要继续运行损坏的模型。

## 4. 创建 Python 环境

以下示例基于 Debian/Ubuntu 系板卡镜像：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgomp1
cd /opt/yolo_gauge_rk3588_python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rk3588.txt
```

本包为 Python 3.8 锁定以下版本，避免 pip 自动选择已经停止支持 Python 3.8 的新版本：

```text
numpy==1.24.4
opencv-python-headless==4.8.1.78
PyYAML>=6,<7
```

`.venv` 在板端创建，不属于迁移包。若设备无法联网，可在另一台相同架构、相同 Python ABI 的机器下载依赖 wheel 后离线安装；不能把 x86_64 wheel 安装到 RK3588。

如果厂商镜像已提供 `python3-opencv`，也可以使用系统 OpenCV，但不要同时混装多个 OpenCV Python 包。服务器镜像推荐 `opencv-python-headless`，无需桌面 GUI。

### 安装 RKNNLite

从板卡厂商 SDK 或 Rockchip RKNN Toolkit2 2.3.2 官方发布包中取得 Python 3.8/aarch64 wheel。当前板端必须使用 `cp38`，不要安装 `cp310`、`cp311` 或 x86_64 wheel：

```text
rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

安装命令：

```bash
python -m pip install \
  /path/to/rknn_toolkit_lite2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

板端需要的是 `rknn-toolkit-lite2`，不是用于 x86 模型转换的 `rknn-toolkit2`。

检查环境：

```bash
python - <<'PY'
import platform
import cv2
import numpy
import yaml
from rknnlite.api import RKNNLite

print("machine:", platform.machine())
print("python:", platform.python_version())
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("RKNNLite:", RKNNLite)
PY
```

期望 `machine` 为 `aarch64`、`python` 为 `3.8.x`、NumPy 为 `1.24.4`、OpenCV 为 `4.8.1`。如果 RKNNLite 能导入但 `init_runtime` 失败，优先检查 Runtime/驱动组合，不要更换 Python 依赖来掩盖问题。

## 5. 运行检测

启动脚本会自动设置 `PYTHONPATH`，不需要安装本项目本身。先增加执行权限：

```bash
chmod +x scripts/run_rknn_detection.sh
```

检测单张图片：

```bash
bash scripts/run_rknn_detection.sh \
  --python .venv/bin/python \
  --input /data/images/detect-01.jpg \
  --output /data/results/single
```

检测目录中的所有图片：

```bash
bash scripts/run_rknn_detection.sh \
  --python .venv/bin/python \
  --input /data/images \
  --output /data/results/all
```

递归检测子目录：

```bash
bash scripts/run_rknn_detection.sh \
  --python .venv/bin/python \
  --input /data/images \
  --output /data/results/all \
  --recursive
```

指定其他模型、配置或 NPU 核心：

```bash
bash scripts/run_rknn_detection.sh \
  --python .venv/bin/python \
  --model /data/models/custom.rknn \
  --config configs/rk3588.yaml \
  --input /data/images \
  --output /data/results \
  --core-mask CORE_0_1_2
```

支持的核心模式为 `AUTO`、`CORE_0`、`CORE_1`、`CORE_2`、`CORE_0_1` 和 `CORE_0_1_2`，默认 `AUTO`。先用 `AUTO` 完成精度验收，再比较其他模式。

每张输入图默认生成：

- `*_annotated.jpg`：带编号检测框的结果图；
- `*.json`：框坐标、置信度和 RKNN Runtime 推理耗时；
- `crops/`：目标裁剪图。

启动脚本的完整参数可通过以下命令查看：

```bash
bash scripts/run_rknn_detection.sh --help
```

## 6. 配置说明

默认配置为 `configs/rk3588.yaml`。当前模型必须保持：

```yaml
model:
  backend: rknn
  rknn_path: models/yoloe-26s-rk3588-fp16.rknn
  input_shape: [544, 960]
```

可以调整 `detection.conf`、`detection.iou`、输出开关和绘框粗细，但每次修改都应重新跑完整测试集。不能改变 `backend`、输入 Shape、颜色顺序或归一化方式来“修复”检测结果。

配置中的多个英文 Prompt 只是模型来源记录；Prompt 已静态固化，板端编辑该列表不会改变模型能力。

## 7. 上板验收

### 正确性

使用与主机基线相同的图片集检测并人工检查标注图。当前项目的关键要求是：

- 每张图最终最多一个框，无重复框和嵌套框；
- `detect-01`、`detect-09`、`detect-10` 应检出；
- 完整仪表框不能被内部表盘小框取代；
- 如具备主机基线坐标，最终框 IoU 建议不低于 0.80。

### 性能

`JSON` 中的 `inference_ms` 只统计 `RKNNLite.inference()`，不包含读图、预处理、Python 解码、绘图和写盘。性能测试应先预热至少 3 次，再对同一批图片运行至少 50 次，记录中位数、P90 和 P99；不要用单次最快值作为结论。

依次比较 `AUTO` 和各 core mask，只有在检测框完全一致时才选择更快的模式。正式运行时保持一个 Python 进程和一个模型实例，不要每张图重新启动脚本。

### 稳定性

用最终核心模式持续循环输入至少 30 分钟，并监控：

- Runtime 错误、超时、RKNPU reset 和 OOM；
- 进程 RSS 是否持续增长；
- SoC/NPU 温度、CPU/NPU 频率和热降频；
- 关键图片是否出现间歇漏检。

如果热降频明显，应先改善散热与供电。降低输入分辨率会改变精度，必须重新导出模型并完成全量回归。

## 8. 常见问题

### `No module named rknnlite`

RKNNLite wheel 未安装到启动脚本所用的 Python。确认 `--python` 指向正确虚拟环境，并执行：

```bash
.venv/bin/python -c 'from rknnlite.api import RKNNLite; print(RKNNLite)'
```

### `wrong ELF class`、`Exec format error` 或 wheel 不受支持

wheel 架构或 Python ABI 不匹配。RK3588 必须使用 `aarch64` wheel；本包的 Python 3.8 对应 `cp38`。`cp310`、`cp311` 和 `x86_64` 均不能用于当前板端环境。

### `load_rknn` 或 `init_runtime` 返回非零错误

通常是模型、RKNNLite、`librknnrt` 和 RKNPU 驱动版本不兼容。保存完整日志并对照板卡厂商版本矩阵，不要通过替换随机来源的系统动态库解决。

### 运行时提示模型不存在

必须从迁移包根目录运行脚本，或通过 `--model` 指定绝对路径。重新执行 `sha256sum -c SHA256SUMS` 检查模型是否完整。

### 无检测或框位置明显错误

先确认没有修改 `input_shape`，输入图片能够由 OpenCV 正常读取，并且预处理仍为 BGR 读图后转 RGB、uint8 NHWC、应用侧不除以 255。不要直接降低阈值掩盖颜色、Shape、padding 或 Runtime 版本错误。

### OpenCV 导入失败

确认只安装了一个 OpenCV Python 包。若系统缺少运行库，可安装 `libgomp1`；桌面版 OpenCV 还可能需要 GUI 库，板端建议改用 `opencv-python-headless`。

## 9. 回滚与更新

迁移包是自包含目录。更新前保留旧目录和旧压缩包，将新版本解压到不同目录，完成校验及回归后再切换启动路径。回滚时只需恢复旧目录，不要随意回滚板端 NPU 驱动或系统 Runtime。

真实 RK3588 NPU 的精度、速度、核心配置和温度稳定性只能在目标板上最终确认；x86 主机测试不能替代板端验收。
