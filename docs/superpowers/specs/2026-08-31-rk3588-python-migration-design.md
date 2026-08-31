# RK3588 Python 迁移设计

## 目标

在不改变当前仪表筛选规则和业务输出格式的前提下，为项目增加可迁移到 RK3588 的 Python 推理链路。所有能够在当前 x86 主机完成的导出、格式检查、一致性测试和回归测试先在主机完成；RK3588 上只保留 RKNNLite Python 推理、预处理、后处理和现有图片输出功能。

本阶段不实现 C++，不替换当前稳定的 PyTorch 默认后端，也不在缺少 RK3588 硬件时宣称板端推理通过。

## 已冻结的基线

- 模型：`yoloe-26s-seg.pt`
- Ultralytics：`8.4.121`
- PyTorch 推理尺寸：`imgsz=960`，使用 Ultralytics 原生最小矩形预处理
- 部署静态输入：`1×3×544×960`，匹配当前固定 1920×1080 图像的原生预处理结果
- Text Prompt：`analog gauge`、`dial gauge`、`pressure gauge`、`pressure meter`、`industrial gauge`
- 检测阈值：`conf=0.15`、`iou=0.50`、`agnostic_nms=true`
- 后处理：每张图最多保留一个目标；最高置信度框出现嵌套时保留所在嵌套组的最大框
- 主机单元测试基线：`29 passed, 1 skipped`
- `/home/mcl/data/yolo/检测采图` 的 11 张图片均输出一个框
- 重点回归结果：
  - `detect-01`：`[371, 440, 612, 657]`，置信度 `0.2569`
  - `detect-09`：`[855, 366, 1093, 555]`，置信度 `0.1705`
  - `detect-10`：`[240, 401, 390, 552]`，置信度 `0.2189`

基线结果保存在 `outputs/rk3588_host_baseline/`。

## 环境边界

当前项目 `.venv` 使用 Python 3.13.5，负责保持现有 PyTorch/Ultralytics 推理稳定。RKNN-Toolkit2 当前不支持 Python 3.13，因此转换工具使用独立的 Python 3.10 环境，不能升级或替换现有 `.venv` 中的核心依赖。

主机端环境职责：

1. 现有 `.venv`：生成 Prompt Embedding、构建纯检测 YOLOE、导出 ONNX、运行 PyTorch/ONNX 一致性测试。
2. 独立 RKNN 转换环境：ONNX 转 RKNN、FP16/INT8 构建、精度分析和 RKNN 模拟器检查。
3. RK3588 板端环境：RKNN-Toolkit-Lite2 Python、OpenCV、NumPy；不安装 PyTorch、Ultralytics或文本编码器。

转换环境、板端 `librknnrt` 和 RKNPU 驱动版本必须记录并匹配。第一版锁定 RKNN-Toolkit2 2.3.2；如果厂商系统镜像自带的 Runtime/驱动不兼容，则以板卡厂商验证过的版本矩阵为准，并重新生成 RKNN 模型。

## 架构

### 离线模型准备

1. 从当前配置加载原始 `yoloe-26s-seg.pt`。
2. 调用 `set_classes()` 初始化相同的 Text Prompt。
3. 保存与 checkpoint 和 Prompt 列表绑定的 NPZ profile，同时保存 JSON 元数据和 SHA256。
4. 从匹配的 `yoloe-26s.yaml` 构建纯检测模型并加载分割 checkpoint 中兼容的权重。
5. 加载同一个 Prompt profile，导出静态、batch=1、固定尺寸的 ONNX。
6. 导出两份静态图：主机 ONNX 保留 end-to-end `1×300×6` 输出；RKNN 源图复用 YOLO26 one-to-one 检测头但关闭不受支持的 TopK，输出解码后的框通道和类别分数。不能直接切换到 one-to-many 头，否则会改变嵌套候选框。
7. 验证两份模型均不包含 mask 业务输出，并记录实际输入、输出张量名称、Shape 和 dtype。

提示词在导出模型中静态固化。修改提示词、模型权重或部署输入 Shape 后必须重新导出 ONNX/RKNN，不支持在 RK3588 运行时调用 `set_classes()`。

### 运行时后端

增加窄接口后端，避免重写上层业务逻辑：

```python
class InferenceBackend:
    imgsz: int
    def predict(self, bgr_image: np.ndarray, *, conf: float, iou: float, max_det: int): ...
```

保留当前 `YOLOEModel` 作为默认 PyTorch 后端。新增 ONNX 后端用于主机一致性验证，新增 RKNNLite 后端用于 RK3588。后端输出统一转换成项目现有 `Detection`，之后继续复用类别合并、去重、几何过滤、单目标选择、从左到右排序、绘框、JSON 和裁剪逻辑。

### 预处理契约

- 输入：非空 OpenCV BGR `uint8`，形状 `H×W×3`
- 颜色：BGR 转 RGB，只转换一次
- Resize：保持长宽比的 letterbox
- 输入尺寸：静态矩形，第一版为高 544、宽 960；该 Shape 对应当前固定 1920×1080 图像在 `imgsz=960` 下按 stride=32 对齐的最小矩形
- Batch：1
- Padding 颜色、缩放比例、左右/上下 padding 必须显式记录
- 归一化只能由应用或 RKNN 配置中的一方完成，不得重复
- 坐标恢复统一使用记录的 ratio 和 padding，并裁剪到原图范围

同一组预处理函数必须供 ONNX 和 RKNNLite 使用。任何与 Ultralytics 输出不一致的情况，先检查颜色、归一化、布局、padding 和输出解码，不通过降低阈值掩盖。

### 后处理契约

不能预设 YOLOE-26 的导出张量与 YOLOv8 相同。主机 ONNX 已实测为 `1×300×6`，格式为 `[x1,y1,x2,y2,confidence,class_id]`。RKNN 使用 YOLO26 one-to-one 头的 `1×(4+类别数)×anchors` 原始输出：前 4 通道是已解码的 `xywh`，后续通道是类别分数，Python 后处理负责转为 `xyxy` 并执行类别无关 NMS。实测直接关闭 end-to-end 并使用默认 one-to-many 头会使 `detect-06` 的最终框 IoU 降至约 0.62，因此禁止该导出方式。后处理完成以下行为：

1. 将多个 Prompt 类别统一映射为 `instrument`。
2. 执行类别无关的候选过滤和必要的 NMS。
3. 复用现有 `remove_duplicate_boxes()`。
4. 复用现有 `select_single_target()`，避免改变已经稳定的嵌套框规则。
5. 保持现有 JSON、绘图和裁剪格式。

## 配置与命令行

新增独立 `configs/rk3588.yaml`，默认仍保留 `configs/default.yaml`。`imgsz` 保留为 Prompt profile 和 PyTorch 配置，`input_shape` 单独描述部署模型的静态 `[height,width]`。配置至少包括：

- `backend`: `pytorch`、`onnx` 或 `rknn`
- 各后端模型路径
- 静态输入尺寸和输入布局
- letterbox padding 颜色
- RKNN target、量化模式、校准数据清单
- RKNNLite core mask，默认 `AUTO`

CLI 在现有 `detect`、`detect-dir` 入口中根据配置选择后端，不新增重复的图片保存流程。模型准备使用独立命令：保存 Prompt profile、导出纯检测 ONNX、转换 RKNN、比较后端结果。

## 分阶段测试门槛

每一阶段严格执行测试先行，并且只有当前阶段通过才进入下一阶段。

### 阶段 1：Prompt profile

- 单元测试先验证 Prompt 列表、checkpoint SHA256 和 NPZ/JSON 绑定关系。
- 测试重复生成具有稳定元数据。
- 实际生成 profile 后，重新加载并在主机上完成一次 PyTorch 推理。

### 阶段 2：纯检测 ONNX

- 单元测试验证模型架构名从 `*-seg.pt` 正确映射到同尺度 detection YAML。
- 测试拒绝 prompt-free checkpoint、错误文件名和动态输入。
- 实际导出后用 ONNX checker 检查模型。
- 使用至少一张图片比较 PyTorch 与 ONNX 候选框。

### 阶段 3：ONNX 后端

- 用手工构造张量分别测试 end-to-end ONNX 输出、RKNN 原始输出、letterbox、坐标恢复和空检测。
- 用实际 ONNX Runtime 跑 11 张图片。
- `detect-01/09/10` 必须检出；最终框与基线 IoU 不低于 0.80。
- 11 张图片不能产生重复业务框。

### 阶段 4：RKNN 转换

- 在没有安装 RKNN 的现有环境中，转换模块仍可导入，并给出明确的依赖错误。
- 测试校准清单生成、配置解析、FP16/INT8 参数映射。
- 先生成 544×960 FP16 RKNN；主机只验证模型生成、元数据和模拟器能力，不将模拟器结果冒充板端结果。
- FP16 通过后才尝试 INT8；INT8 校准集必须包含关键图和负样本。

### 阶段 5：RKNNLite Python 后端

- 使用依赖注入的假 Runtime 测试初始化一次、输入设置、原始输出解码、输出读取、释放和错误传播。
- RK3588 上再运行真实 RKNNLite 集成测试。
- 第一轮只用 `AUTO` core mask；板端正确性通过后再比较单核、双核和三核。

### 阶段 6：全量回归

- 全部单元测试通过。
- PyTorch 默认配置的 11 张图片结果不得回退。
- ONNX/RKNN 主机可执行部分输出独立报告，包含预处理、推理、后处理和端到端耗时。
- 960 通过后，才按 768、640 顺序实验；任何尺寸造成 `detect-09/10` 漏检即淘汰。

## 错误处理

- 模型、Prompt profile 或配置缺失时，在初始化阶段失败。
- Prompt 元数据与 checkpoint、Prompt 列表或输入尺寸不匹配时拒绝导出。
- ONNX/RKNN 输出 Shape 不受支持时报告所有实际张量信息，不猜测解码格式。
- 主机缺少 RKNN/ONNX 可选依赖时给出对应环境和安装命令，不影响 PyTorch 默认后端。
- RK3588 Runtime/驱动不兼容时输出版本信息并停止，不自动替换系统库。

## 非目标

- 本阶段不实现 C++、摄像头 DMA/RGA 零拷贝和多线程流水线。
- 不训练闭集单类别 YOLO。
- 不默认切换到 YOLOE-26n、768 或 640。
- 不在缺少板卡的主机上承诺 RK3588 延迟或吞吐量。
- 不改变现有单目标和嵌套框选择语义。

## 完成标准

主机端完成的定义是：静态 Prompt profile、纯检测 ONNX、ONNX 后端、RKNN 转换工具、RKNNLite Python 后端及测试全部落地；现有 PyTorch 回归通过；11 张关键图完成可用后端的一致性测试；生成板端测试说明和版本清单。真实 RKNN 推理正确性、NPU 延迟、多核效果和长时间温度稳定性必须在 RK3588 到位后另行验收。
