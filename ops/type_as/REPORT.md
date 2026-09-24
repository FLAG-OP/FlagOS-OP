# 算子总体报告: type_as

| 项 | 值 |
|---|---|
| 算子名称 | `type_as`（`self` cast 到 `other.dtype`，留在 `self.device`） |
| 语义 | [reference.py](reference.py)（CPU 原生 `.to`，位级判卷标准） |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu / PrivateUse1） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1**（aten `type_as` 拦截） |
| 日期 / 状态 | 2026-09-18 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | ATen `.to`（原生对照） |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ | 自研 strided cast，含 #11 device 上下文 |
| 硬件级 | [kernel/hardware_level/](kernel/hardware_level/README.md) | ⬜ 置空 | 本机无同语义厂商 cast 原语 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细报告 |
|---|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ | — |
| kernel 层 | `python3 test/kernel_level.py cambricon` | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `python3 test/op_level.py cambricon` | ✅ | — |
| 应用层 | `python3 test/framework_level.py cambricon` | ✅（等价 torch API，vLLM 未安装） | — |
| 黄金回归 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 111/111 | [goldendata/](goldendata/README.md) |
| 性能 | `scripts/perf_run.py` / 基线门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 自研 Triton | 原生 ATen | FlagGems `to_copy` |
|---|---|---|---|
| 精度（111 组黄金） | **0**（位级一致） | 0 | 0（抽样） |
| 8192² fp16→fp32 延迟 | 0.190 ms | **0.175 ms** | 0.317 ms |
| 哨兵 | 确定性 ✅ 敏感 ✅ | — | — |
| 应用层命中 / 输出一致 | 1 次 / 逐位一致 | — | — |

## 结论与遗留

三层全绿、黄金 111/111、精度位级一致；性能比原生慢约 9%，但比
FlagGems `to_copy` 快约 40%。遗留: 真实 vLLM 应用层（环境未装）、
连续 cast 性能优化、非 dense 窄→宽的两阶段 kernel（当前用
`out.copy_` 兜底，见 [reports/development.md](reports/development.md) §6/§7）。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告（7 章） | [reports/development.md](reports/development.md) |
| 测试报告（范围矩阵） | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
