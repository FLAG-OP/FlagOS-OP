# 算子测试报告: &lt;算子名&gt;

| 项 | 值 |
|---|---|
| 算子名称 | `<op_name>` |
| 实现路线 | A1 / A2 / B（勾选） |
| 开发级别 | torch 级 / Triton 级 / 硬件级 |
| 目标设备 | `<device profile>` |
| 测试日期 | `<YYYY-MM-DD>` |
| 结论 | **通过 / 有条件通过 / 不通过** |

## 1. 测试范围

| 层级 | 是否覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ☐ | `run.py --level kernel` | |
| 框架层（注册/钉选/拦截） | ☐ | `run.py --level op` | |
| 应用层（真实推理） | ☐ | `run.py --level framework` | |
| 跨层一致性 | ☐ | `run.py --consistency` | |
| 性能回归门禁 | ☐ | `scripts/perf_compare.py` | |

## 2. 环境

> `python3 scripts/env_snapshot.py --device <profile>` 自动生成；
> `python3 scripts/check_env.py --device <profile>` 核对镜像版本。

## 3. 精度结果

| dtype | 容差 | max err | 判定 | 对照 FlagGems | 对照原生 |
|---|---|---|---|---|---|
| bf16 | | | | | |
| fp16 | | | | | |
| fp32 | | | | | |

> 口径: pointwise 用 abs；带乘法放大的融合算子与 GEMM 用相对误差
> （`scripts/accuracy_report.py` 同款）。哨兵: 确定性 ☐ 输入敏感 ☐。

## 4. 性能结果

| 实现 | 延迟 | 带宽/算力 | 相对自研 |
|---|---|---|---|
| 自研 | | | 1.00x |
| PyTorch 原生 | | | |
| FlagGems | | | |
| 厂商（如有） | | | |

> 短采样（≤100 次）+ 子进程隔离采集；对照 [性能基线](../docs/performance-regression.md)。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | | | |

## 6. 结论

<一段话: 是否可合入、遗留项、后续优化方向>
