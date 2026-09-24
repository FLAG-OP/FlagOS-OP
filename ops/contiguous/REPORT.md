# 算子总体报告: contiguous

| 项 | 值 |
|---|---|
| 算子名称 | `contiguous` |
| 语义 | [reference.py](reference.py)（原生 `.contiguous`，位级标准） |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu / PrivateUse1） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ | — |
| kernel 层 | `test/kernel_level.py` | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ✅ | — |
| 应用层 | `test/framework_level.py` | ✅（等价 torch API） | — |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 114/114 | [goldendata/](goldendata/README.md) |
| 性能 | perf 门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 自研 Triton | 原生 |
|---|---|---|
| 精度（114 组） | **0**（位级一致） | 0 |
| 8192² fp32 转置→连续 | 0.305 ms | **0.254 ms** |
| 别名 / 连续性 / 哨兵 | ✅ / ✅ / ✅ | — |

## 结论与遗留

三层全绿、黄金 114/114、别名语义正确。转置性能经 2D swiz 分块修复，
从 21 GB/s 提升到 1762 GB/s（~80x），仍比原生慢 ~20%。
遗留: rank≥3 分块、channels_last 自研、真实 vLLM 应用层。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
