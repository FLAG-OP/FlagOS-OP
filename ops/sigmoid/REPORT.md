# 算子总体报告：aten.sigmoid

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 算子 | `aten::sigmoid(Tensor self) -> Tensor` |
| 语义 | `1 / (1 + exp(-x))`；见 [reference.py](reference.py) |
| 路线 / 级别 | A1 / Triton |
| 设备 | 已提交 profile `p800-kunlunxin-gems5`；物理 XPU 1/2，逻辑 `cuda:0/1`，算子层 `cuda:1`，vLLM TP=2 |
| 日期 / 状态 | 2026-11-27 / 当前 P800 非 C2 验收完成；真实 C2 受 NCCL 初始化阻塞 |

## 实现矩阵

| 级别 | 状态 | 文件 |
|---|---|---|
| torch | 完成 | [kernel/torch_level.py](kernel/torch_level.py) |
| Triton | 实现完成；当前 P800 三层、黄金和性能复测完成 | [kernel/triton_level.py](kernel/triton_level.py) |
| 硬件级 | 未实现，本次范围外 | [kernel/hardware_level/BUILD.md](kernel/hardware_level/BUILD.md) |

## 验证矩阵

| 层级 | 当前状态 | 关键证据 |
|---|---|---|
| kernel | PASS | 当前提交 75 例、19 shape，max abs/rel=0，确定性与敏感性通过 |
| A1 op | PASS | 当前提交 `AutogradCUDA+CUDA` 注册；拦截 9 次，契约计数 9 次 |
| C1 应用 | PASS | 当前提交 `2×64×4096` 的 fp16/bf16/fp32 均命中且输出一致 |
| C2 vLLM | FAIL（环境初始化） | baseline TP worker 在 `ncclCommInitRank` 报 NCCL error，未进入生成/注入；见[可复核材料](evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md) |
| 黄金 | PASS | 当前 Triton 93/93；最差 abs `1.192e-07`、rel `1.680e-07` |
| 性能矩阵 | 已采集 | 当前 36/36；自研 0/12 快于原生，原始 JSON 已入库 |
| dispatch | 已采集 | 直调/注册 `54.655/59.369 μs`，增量 `4.714 μs`（8.6%） |
| 中央性能入口 | 无有效回归门禁 | 历史 `NEW + 0 FAIL` 无 baseline，只表示成功入表 |

## 当前性能摘要

- 当前 FlagGems 5.0 环境中，自研 12 个组合均未超过原生。
- 最接近原生的是 `2048×2048 fp32`：自研 `0.058315 ms`、原生 `0.048688 ms`，比值 `1.198×`。
- 最差为 `64×1024 fp16` 的 `5.736×`；小/中尺寸仍受约 49～53 μs 固定启动成本影响。
- 原始 36 项 JSON 与日志见[当前验收材料](evidence/p800-current-validation-2026-11-27/ANALYSIS.md)，默认路由继续使用 native。

## 结论

2026-11-27 当前 PR 已通过 200 项 CPU 单测、证据检查、P800 三层测试、Triton 黄金 93/93，并完成 36 项性能矩阵与 dispatch 采集。当前非 C2 硬件验收已闭环，原始日志、JSON、命令、提交与镜像摘要见[当前验收材料](evidence/p800-current-validation-2026-11-27/ANALYSIS.md)。真实 C2 baseline 仍在 TP worker 的 `ncclCommInitRank` 阶段因 NCCL 错误失败，未进入生成或 sigmoid 注入，因此 C2 不能标记为通过；完整故障材料见[运行时故障材料](evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md)。性能上自研没有超过原生，默认路由保持 native。

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 应用范围 | [reports/scope.md](reports/scope.md) |
