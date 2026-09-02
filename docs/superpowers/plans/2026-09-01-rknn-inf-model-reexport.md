# RKNN Inf 模型重新导出与验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不覆盖现有 PyTorch CPU 方案的前提下，重新导出 RKNN 模型并验证 INT8/FP16 方案是否消除板端 Inf 输出。

**Architecture:** 先封存板端当前 PyTorch 目录，再在主机使用既有 ONNX/profile 和 RKNN Toolkit2 导出候选模型。每个候选模型单独复制到板端临时目录，用固定输入执行原始输出有限值统计和实际检测；失败模型保留日志，不替换当前可运行方案。

**Tech Stack:** Python 3.11 主机、RKNN Toolkit2、ONNX、RKNNLite Python 3.8、RK3588 NPU。

**Spec:** `docs/superpowers/specs/2026-08-31-rk3588-python-migration-design.md`

## Global Constraints

- 不删除或覆盖 `~/yolo/yolo_gauge_pytorch_cpu` 和现有 RKNN 项目。
- 每次板端目录/模型/环境写入前先审核。
- 先尝试 INT8；若 INT8 不可行，再尝试 FP16 导出参数调整。
- 每个候选模型必须记录加载结果、输出 shape、NaN/Inf 数量及单图检测结果。

### Task 1: 封存当前 PyTorch CPU 方案

**Files:** 板端 `~/yolo/yolo_gauge_pytorch_cpu`；新增压缩包 `~/yolo/archive/`。

- [ ] 创建带时间戳的 tar.gz 封存包，不删除原目录。
- [ ] 校验压缩包可列出项目源码、模型和配置；记录 SHA256。

### Task 2: 检查主机导出输入资产

**Files:** 只读检查 `artifacts/rk3588/`、`profiles/`、`configs/onnx.yaml`。

- [ ] 确认 ONNX、prompt profile 和 RKNN Toolkit2 版本。
- [ ] 若缺少导出输入，停止并报告，不修改板端环境。

### Task 3: 导出 INT8 候选模型

**Files:** 新建主机 `artifacts/rk3588/candidates/int8/`。

- [ ] 使用现有转换脚本和校准图片生成 INT8 RKNN。
- [ ] 记录转换日志和模型 SHA256；不覆盖当前 FP16 模型。

### Task 4: 板端验证 INT8 候选

**Files:** 板端新目录 `~/yolo/rknn_candidates/int8/`、结果日志。

- [ ] 复制候选模型和元数据。
- [ ] 执行固定输入 raw output 有限值统计及单图检测。
- [ ] 只有输出全为 finite 且检测成功时才考虑后续精度测试。

### Task 5: FP16 备选导出与验证

- [ ] 仅当 INT8 失败或精度不可接受时执行。
- [ ] 尝试关闭混合精度/易溢出算子相关导出选项，生成独立候选并重复 Task 4。

### Task 6: 汇总结论

- [ ] 对比候选模型 Inf/NaN、推理耗时和检测框。
- [ ] 推荐可行模型；失败时保留现状并说明需升级驱动或重新设计模型。
