# 多平台合并流程（MERGE）— empty_strided

> 新平台实现合入本目录的固定流程。本算子无独立 `kernel/<platform>.py`——
> torch 级实现平台无关；Triton 级按 `device.type` 自适应。核心原则：
> **黄金/测试/注册链复用，只改设备相关分支**。

## 七步流程

1. **黄金复用**：`goldendata/` 平台无关（CPU 生成）；
   `python3 script/check_accuracy.py --impl <impl> --device <dev>` 过黄金
2. **设备实现**：torch 级 `kernel/torch_level.py`：设备无关 stride 分配
3. **register 选择器**：`register.py` 的 `_BACKENDS` 加平台、
   `_PROBE_MODULE`/`_PROBE_DEVICE` 加探测、`_a1_keys` 成对键覆盖
4. **设备能力差异**：写入 `PLATFORM.md`（channels_last / empty_like preserve /
   工厂算子 dispatch 等）
5. **排序陷阱**：`TASK_QUEUE_ENABLE=0` 须早于 torch_npu 初始化
   （`common/device.py` 已强制；见 `docs/known-issues.md` #24）
6. **三层验证**：`python3 example.py <profile>` 三层全绿
7. **文档同步**：`ops/catalog.yaml` 平台列、`PLATFORM.md`、`reports/`

## 验证矩阵终态（ascend910）

| 层 | 命令 | 结果 |
|---|---|---|
| kernel | `python3 test/kernel_level.py ascend910` | checks=16 全过 |
| op | `python3 test/op_level.py ascend910` | 拦截 2 次 |
| 应用 | `python3 example.py ascend910` | 三层全绿（应用层=C1 等价消费方子进程；真实 vLLM 受 Ascend 内部重入限制，见 [#26](../../docs/known-issues.md#26)） |
| 黄金 | `python3 script/check_accuracy.py --impl torch/triton --device npu` | 30/30 |
| 性能 | `python3 scripts/perf_run.py --device ascend910 --pattern ops.empty_strided.` | 入库 |

---
Dashener2 2026-10-15-fix
# Dashener2 <dashen272@outlook.com>
