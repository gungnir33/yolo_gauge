# RK3588 Python Host Validation

验证日期：2026-08-31

分支：`feat/rk3588-python`

主机：Intel Core Ultra 9 285K，CPU 推理

## 环境与模型

- Python/PyTorch：3.13.5 / 2.13.0+cpu
- Ultralytics：8.4.121
- ONNX/ONNX Runtime：1.18.0 / 1.29.0
- RKNN 转换环境：Python 3.10.12，RKNN-Toolkit2 2.3.2
- Prompt/Profile `imgsz`：960
- 部署静态输入：`1×3×544×960`
- 主机 ONNX 输出：`1×300×6 float32`
- RKNN 源输出：one-to-one raw `1×9×10710 float32`
- FP16 RKNN 大小：约 22 MB

本次本地生成物 SHA256：

```text
7371c9c6906d523c44c6169e227ff7865e186949845ab557f3ba2a456af0dafa  yoloe-26s.onnx
54c07ce97a1ed5bfb2b1b5920ede0d2ab9568dbb703d6f9dc42d09ea4853a6e2  yoloe-26s-rknn-source.onnx
bfd6f0cb2a7af95ce4a26a164310211985d9969ee0ab70208953bce0a18749fc  yoloe-26s-rk3588-fp16.rknn
0307286ba9d32d3a41e94e66f778784de3cbd84d7a073ef1e2c10829b5f85168  gauge-prompts.npz
```

## 测试结果

- 单元测试：91 passed，1 skipped（硬件集成默认跳过）。
- 真实模型加载集成测试：1 passed。
- 默认 PyTorch 11 图：11/11 检出，每图一个框，所有框相对冻结基线 IoU=1.0000。
- 主机 end-to-end ONNX 11 图：11/11 检出，无重复框，所有最终框相对冻结基线 IoU=1.0000。
- one-to-one raw ONNX + Python decoder：11/11 检出；最小 IoU=0.9893，平均 IoU=0.9948，四舍五入后的 JSON 框与基线一致。
- 关键图 `detect-01/09/10` 均通过。

曾验证 960×960 静态输入会改变 `detect-02` 的候选排序，因此改为与原生最小矩形一致的 544×960。直接使用 one-to-many raw 头会使 `detect-06` 的嵌套框 IoU 降至约 0.62，因此 RKNN 源图固定使用 one-to-one raw 头。

## 主机时延

每个后端预热 3 次。ONNX 每图运行 5 次并取阶段中位数，再对 11 图求平均；图片读取不计入。PyTorch 每图运行 3 次取中位数。

| Backend | Preprocess ms | Runtime/Inference ms | Project postprocess ms | End-to-end ms |
|---|---:|---:|---:|---:|
| PyTorch | Ultralytics 内部 | 71.55 | 0.08 | 71.63 |
| ONNX Runtime | 2.55 | 25.86 | 0.11 | 29.94 |

ONNX 逐图中位数：

| Image | Pre | Inference | Post | End-to-end |
|---|---:|---:|---:|---:|
| detect-01 | 1.99 | 26.10 | 0.11 | 28.70 |
| detect-02 | 4.25 | 25.99 | 0.11 | 43.74 |
| detect-03 | 1.80 | 23.86 | 0.10 | 26.63 |
| detect-04 | 1.81 | 24.27 | 0.11 | 27.70 |
| detect-05 | 3.99 | 27.65 | 0.10 | 31.03 |
| detect-06 | 1.73 | 26.34 | 0.12 | 28.69 |
| detect-07 | 1.84 | 25.74 | 0.11 | 27.63 |
| detect-08 | 3.51 | 25.01 | 0.11 | 27.58 |
| detect-09 | 1.79 | 24.70 | 0.11 | 27.37 |
| detect-10 | 1.82 | 25.78 | 0.10 | 28.13 |
| detect-11 | 3.52 | 29.03 | 0.11 | 32.08 |

## RKNN 主机验证

- FP16 build 成功并生成 `.rknn` 与 JSON 元数据。
- Toolkit 构建日志出现 3 条 `Unknown op target: 0`，但 build 返回 0。
- 从 ONNX build 后启动 RKNN simulator 成功，输出 `(1,9,10710) float32`。
- 模拟器 `detect-06` 输出置信度 0.7012，最终框 `[801,612,909,718]`，与基线一致。
- 已导出的 `.rknn` 不能在无 target 的主机 simulator 通过 `load_rknn()` 直接运行；这是 Toolkit 明确限制，不是模型加载失败。

## 板端状态

以下均为 `BOARD_REQUIRED`：真实 NPU 精度、NPU 延迟、core-mask 对比、Runtime/驱动兼容性和 30 分钟热稳定性。主机报告不将模拟器结果标记为板端通过。
