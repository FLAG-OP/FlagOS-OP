# 算子开发报告: &lt;算子名&gt;

> 7 章模板。已填范本（真实数据）:
> [softmax-fullstack/reports/development.md](../../../examples/softmax-fullstack/reports/development.md)

| 项 | 值 |
|---|---|
| 算子名称 | `<op_name>` |
| 实现路线 / 开发级别 | A1 / A2 / B · torch 级 / Triton 级 / 硬件级 |
| 目标设备 | `<device profile>` |
| 日期 / 状态 | `<YYYY-MM-DD>` / 草稿·评审·定稿 |

## 1. 环境配置

> `python3 scripts/check_env.py --device <profile>` 输出 + 设备 profile
> 摘要（格式见范本第 1 章）。

## 2. 算子定义

语义（指向 reference.py）/ 接口签名 / 数值规格（容差按
[验收标准](../../../docs/acceptance.md#accuracy) 分档）/ shape 清单
（含非整倍数维度）。

## 3. 实现说明

路线选择依据（[集成指南](../../../docs/integration.md)）/ kernel 设计
要点 / 三级实现状态表（未实现的置空并说明）。

## 4. 验证结果

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel 层 | | 组数 / 最差 err / 哨兵 |
| 框架层 | | 拦截/钉选证据 |
| 应用层 | | 触发次数 / 输出一致性 |
| 黄金回归 | | n/n |

## 5. 性能

三方表（自研/原生/FlagGems）+ 相对值 + **精度代价列**（口径见
[验收标准](../../../docs/acceptance.md#performance)）。

## 6. 已知问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|

## 7. 结论与后续

<可否交付（对照 [Must 清单](../../../docs/acceptance.md#levels)）/
遗留 / 优化方向>

## 附录: 复现命令
