# 算子总体报告: item

| 项 | 值 |
|---|---|
| 算子名称 | `item`（host 标量类） |
| 语义 | [reference.py](reference.py)：单元素张量取为 Python 标量 (device→host 同步) |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu / PrivateUse1） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | — | ⬜ 不适用（无设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 12 项 |
| 框架层 | `test/op_level.py` | ✅ aten::item@PrivateUse1（拦截 1 次） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 36/36 |
| 性能 | 采样 | ✅ 0.0433ms |

## 结论

三层全绿；标量值相等 + Python 类型 + 多元素报错。无 Triton/硬件级（该类算子固有）。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
