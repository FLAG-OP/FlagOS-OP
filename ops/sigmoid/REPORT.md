# 算子总体报告：aten.sigmoid

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`；见 [reference.py](reference.py) |
| 路线 / 级别 | A1 / Triton |
| 设备 | 已提交 profile `p800-kunlunxin-gems5`；物理 XPU 1/2，逻辑 `cuda:0/1`，算子层 `cuda:1`，vLLM TP=2 |
| 日期 / 状态 | 2026-11-27 / CPU 整改完成，当前硬件全量复测与真实 C2 未完成 |

## 实现矩阵

| 级别 | 状态 | 文件 |
|---|---|---|
| torch | 完成 | [kernel/torch_level.py](kernel/torch_level.py) |
| Triton | 实现完成；2026-11-09 历史真机结果保留，当前提交待全量复测 | [kernel/triton_level.py](kernel/triton_level.py) |
| 硬件级 | 未实现，本次范围外 | [kernel/hardware_level/BUILD.md](kernel/hardware_level/BUILD.md) |

## 验证矩阵

| 层级 | 当前状态 | 关键证据 |
|---|---|---|
| kernel | 待当前提交复测 | 75 例为 2026-11-09 修改前历史 PASS |
| A1 op | 待当前提交复测 | 9 次命中为 2026-11-09 修改前历史 PASS；计数器 CPU 契约已修复 |
| C1 应用 | 待当前提交复测 | 3/3 dtype 为 2026-11-09 修改前历史 PASS |
| C2 vLLM | FAIL（环境初始化） | 2026-11-27 baseline TP worker 在 `ncclCommInitRank` 报 NCCL error，未进入生成/注入；见[可复核材料](evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md) |
| 黄金 | CPU 当前通过 / Triton 待复测 | CPU reference/torch 93/93；历史硬件 93/93 不作为当前结论 |
| 性能矩阵 | 待当前提交复测 | 36/36 为 2026-11-09 历史采集，原始 JSON 未入库且不可访问 |
| dispatch | 待当前提交复测 | `47.970/55.877 μs` 为 2026-11-09 历史摘要 |
| 中央性能入口 | 无有效回归门禁 | 历史 `NEW + 0 FAIL` 无 baseline，只表示成功入表 |

## 2026-11-09 历史性能摘要（不可独立复核）

- 正式 FlagGems 5.0 环境中，自研 12 个组合均未超过原生。
- 最接近原生的是 `2048×2048 bf16`：自研 `0.077650 ms`、原生 `0.067039 ms`，比值 `1.158×`。
- 小尺寸受约 49～58 μs 自研启动固定成本影响，最差为 `1×14336 bf16` 的 `4.276×`。
- 自研仅在大张量 bf16/fp32 两项略快于 FlagGems 5.0，原生仍是当前最佳基线。

上述性能原始 JSON 仅曾存放在旧服务器 `/data2/liyanz/results/`，未提交
到仓库且现已不可访问。因此这些数字不构成当前提交的可复核性能证据，
默认路由继续使用 native，直至当前代码重新采集矩阵并通过独立复测。

## 结论

表中 75/9/3、硬件 93/93、36/36 均为 2026-11-09 修改前历史基线。2026-11-27 当前 PR 已通过 200 项 CPU 单测、证据检查和 CPU 黄金两路径 93/93；分阶段探测中基础 copy、native sigmoid 及仓库 sigmoid Triton 的 257/255 最小用例通过，真实 C2 baseline 则在 TP worker 的 `ncclCommInitRank` 阶段因 NCCL 错误失败，未进入生成或 sigmoid 注入。探针中的 generic Triton 项因自身 `tl` 作用域错误无效，不归因于平台。修改后的三层、Triton 黄金、C2 与性能矩阵仍待复测，因此当前只能继续代码评审，不能标记为最终硬件验收完成。完整原始日志、哈希与归因见[运行时故障材料](evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md)。

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 应用范围 | [reports/scope.md](reports/scope.md) |
