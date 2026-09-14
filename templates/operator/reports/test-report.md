# 算子测试报告: &lt;算子名&gt;

> 已填范本（真实数据）:
> [softmax-fullstack/reports/test-report.md](../../../examples/softmax-fullstack/reports/test-report.md)

| 项 | 值 |
|---|---|
| 算子名称 / 路线 / 级别 | |
| 目标设备 / 测试日期 | |
| 结论 | 通过 / 有条件通过 / 不通过 |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ☐ | `test/kernel_level.py` | |
| 框架层 | ☐ | `test/op_level.py` | |
| 应用层 | ☐ | `test/framework_level.py` | |
| 黄金回归 | ☐ | `script/` 两脚本 | |
| 性能门禁 | ☐ | `scripts/perf_compare.py` | |
| 跨层一致性 | ☐ | 仓库 `--consistency` | |

一键: `python3 example.py <profile>`

## 2. 环境

> `check_env` 核对结果（n/n）。

## 3. 精度结果

三方表（dtype × 自研/FlagGems/原生，判定 ✓/⚠）+ 哨兵 + 黄金 n/n。
口径见[验收标准](../../../docs/acceptance.md#accuracy)。

## 4. 性能结果

三方表 + 精度代价列；微算子附 `bench_dispatch.py` 分发开销。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|

## 6. 结论

<对照 Must/Should 分级给结论>
