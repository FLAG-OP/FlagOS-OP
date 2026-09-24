# 算子总体报告: dropout

| 项 | 值 |
|---|---|
| 算子名称 | `dropout` |
| 语义 | [reference.py](reference.py)（原生 F.dropout；随机分支按结构/统计判定） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ Philox |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ | — |
| kernel 层 | `test/kernel_level.py` | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ✅ | — |
| 应用层 | `test/framework_level.py` | ✅（等价 torch API） | — |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 189/189 | [goldendata/](goldendata/README.md) |
| 性能 | perf 门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 自研 Triton | FlagGems | 原生 |
|---|---|---|---|
| 确定性分支精度 | 位级一致 | — | 0 |
| 随机 drop 偏差 | 0.0016 | — | — |
| 8192² fp16 p=0.5 延迟 | 2.785 ms | 2.960 ms | **0.563 ms** |
| seed 可控 / 哨兵 | ✅ / ✅ | — | — |

## 结论与遗留

三层全绿、黄金 189/189；确定性分支位级一致、随机分支统计合规、
`torch.manual_seed` 可控。性能受 Philox RNG 限制，比原生慢 ~5x，
但与 FlagGems 同级（略快）。遗留: 真实 vLLM 应用层、RNG 向量化调优。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
