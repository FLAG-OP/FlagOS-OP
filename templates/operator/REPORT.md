# 算子总体报告: &lt;算子名&gt;

> 定位: 一页看全一个算子——实现矩阵、验证矩阵、关键数字，
> 指向各详细报告。分册在 [reports/](reports/)。

| 项 | 值 |
|---|---|
| 算子名称 | `<op_name>` |
| 语义 | `<一句话 + 指向 reference.py>` |
| 目标设备 / 路线 | `<profile>` / A1 ☐ A2 ☐ B ☐ |
| 日期 / 状态 | `<YYYY-MM-DD>` / 草稿·评审·定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ☐ | |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ☐ | |
| 硬件级 | [kernel/hardware_level/](kernel/hardware_level/BUILD.md) | ☐ | CUDA ✗#12 / 厂商绑定 / SDK |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细报告 |
|---|---|---|---|
| kernel 层 | `python3 test/kernel_level.py --device <profile>` | | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `python3 test/op_level.py` | | |
| 应用层 | `python3 test/framework_level.py` | | |
| 黄金回归 | `script/gen_golden.py` + `check_accuracy.py` | | [goldendata/](goldendata/README.md) |
| 性能 | `script/bench_perf.py` / perf 门禁 | | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 值 | 对照 |
|---|---|---|
| 精度最差 rel err | | 自研/FlagGems/原生 |
| 延迟 | | 三方 |
| 哨兵 | 确定性 ☐ 敏感 ☐ | |

## 结论与遗留

<可否合入 / 风险 / 后续优化>
