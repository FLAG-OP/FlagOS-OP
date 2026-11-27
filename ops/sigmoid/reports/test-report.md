# aten.sigmoid 测试报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / Triton |
| 目标 / 日期 | `p800-kunlunxin-gems5`，物理 XPU 1/2，TP=2 / 当前状态 2026-11-27 |
| 环境 | 官方 FlagGems 5.0 软件锁 |
| 结论 | 当前提交 P800 三层与黄金通过、性能已采集；C2 受 NCCL 初始化阻塞 |

## 1. 覆盖与结果

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel | 19 shape、3 优化 dtype、3 回退 dtype、布局、IEEE、哨兵 | 当前 PASS 75 例；max abs/rel=0 |
| A1 op | 包态注册、counter、非连续/空张量和回退 | 当前 PASS；9 次命中、契约计数 9 次 |
| C1 应用 | `2×64×4096` Transformer 激活，3 dtype | 当前 PASS 3/3，均命中且输出一致 |
| C2 vLLM | 真实 EngineCore 等价消费者双跑 | 2026-11-27 baseline worker 在 `ncclCommInitRank` 报 NCCL error，未进入生成；见[故障材料](../evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md) |
| 黄金 | 边界、长尾、Transformer、饱和值 | 当前 Triton 93/93 |
| 性能矩阵 | 4 shape × 3 dtype × 3 实现，20+100 | 当前 36/36 已采集，JSON 已入库 |
| 中央门禁 | `ops.sigmoid.triton` | 无 baseline；历史 1 NEW + 0 FAIL 不构成门禁通过 |
| dispatch | 直调与 A1 注册对照 | 当前增量 `4.714 μs`（8.6%） |

## 2. 精度

当前 kernel 层自研与参考逐位一致，max abs/rel 均为 0。当前黄金回归最差 abs `1.192e-07`、rel `1.680e-07`，低于 float32 `2e-6` 容差；float16/bfloat16 均逐位一致。NaN mask、空张量、确定性和输入敏感性通过。

## 3. 性能

当前 5.0 矩阵中自研/原生比值为 `1.198–5.736×`，没有组合快于原生。最接近原生的是 `2048×2048 fp32`；小尺寸主要受固定启动成本限制。性能结论不影响功能验收，但默认应用路由继续使用 native。

## 4. 环境注意

只使用已提交的 `p800-kunlunxin-gems5`：容器映射 `/dev/xpu1`、
`/dev/xpu2`，并设置 `XPU_VISIBLE_DEVICES=1,2` 与
`CUDA_VISIBLE_DEVICES=1,2`。可见顺序中物理 XPU 1/2 对应逻辑
`cuda:0/1`；算子层设备为 `cuda:1`，真实 vLLM 为 TP=2、`mp` executor。

## 5. 验收清单

| Must 项 | 结果 | 证据 |
|---|---|---|
| 精度必测矩阵 | 通过 | 当前 kernel 75 例、max abs/rel=0 |
| 确定性和敏感性 | 通过 | 当前 kernel 哨兵通过 |
| 黄金回归 100% | 通过 | 当前 Triton 93/93 |
| 三层测试 | 通过 | 当前 75 / 9 / 3 均通过 |
| 性能与 dispatch | 已采集 | 当前 36/36 与 dispatch 原始产物已入库；中央仍无 baseline |
| 报告与目录 | 完成 | REPORT、五份分册、catalog、生成索引已统一状态 |

## 6. 结论

2026-11-27 当前 PR 已通过 200 项 CPU 单测、证据检查、P800 三层和 Triton 黄金，并完成性能与 dispatch 采集；非 C2 验收原始材料见[当前验收材料](../evidence/p800-current-validation-2026-11-27/ANALYSIS.md)。真实 C2 baseline 仍在 TP worker 的 `ncclCommInitRank` 阶段因 NCCL 错误退出，故 C2 未通过，材料见[运行时故障记录](../evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md)。当前不声明普遍加速、其他平台支持或 backward。
