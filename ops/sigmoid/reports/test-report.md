# aten.sigmoid 测试报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / Triton |
| 目标 / 日期 | `p800-kunlunxin-gems5`，物理 XPU 1/2，TP=2 / 当前状态 2026-11-27 |
| 环境 | 官方 FlagGems 5.0 软件锁 |
| 结论 | 2026-11-09 历史 C1 基线保留；当前提交 CPU 通过、硬件最终复测与 C2 未完成 |

## 1. 覆盖与结果

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel | 19 shape、3 优化 dtype、3 回退 dtype、布局、IEEE、哨兵 | 当前待硬件复测；2026-11-09 历史 PASS 75 例 |
| A1 op | 包态注册、counter、非连续/空张量和回退 | 当前待硬件复测；历史 PASS 9 次命中 |
| C1 应用 | `2×64×4096` Transformer 激活，3 dtype | 当前待硬件复测；历史 PASS 3/3 |
| C2 vLLM | 真实 EngineCore 等价消费者双跑 | 2026-11-27 baseline worker 在 `ncclCommInitRank` 报 NCCL error，未进入生成；见[故障材料](../evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md) |
| 黄金 | 边界、长尾、Transformer、饱和值 | CPU 当前 93/93；硬件 93/93 为历史、当前待复测 |
| 性能矩阵 | 4 shape × 3 dtype × 3 实现，20+100 | 当前待复测；历史 36/36 原始产物不可访问 |
| 中央门禁 | `ops.sigmoid.triton` | 无 baseline；历史 1 NEW + 0 FAIL 不构成门禁通过 |
| dispatch | 直调与 A1 注册对照 | 当前待复测；历史增量 `7.908 μs` |

## 2. 精度

kernel 层自研与参考逐位一致，max abs/rel 均为 0。黄金回归最差 abs `1.192e-07`、rel `1.680e-07`，低于 float32 `2e-6` 容差；float16/bfloat16 均逐位一致。NaN mask、空张量、确定性和输入敏感性通过。

## 3. 性能

正式 5.0 矩阵中自研/原生比值为 `1.158–4.276×`，当前没有组合快于原生。最接近原生的是 `2048×2048 bf16`；小尺寸主要受固定启动成本限制。性能结论不影响功能验收，但限制默认应用范围。

## 4. 环境注意

只使用已提交的 `p800-kunlunxin-gems5`：容器映射 `/dev/xpu1`、
`/dev/xpu2`，并设置 `XPU_VISIBLE_DEVICES=1,2` 与
`CUDA_VISIBLE_DEVICES=1,2`。可见顺序中物理 XPU 1/2 对应逻辑
`cuda:0/1`；算子层设备为 `cuda:1`，真实 vLLM 为 TP=2、`mp` executor。

## 5. 验收清单

| Must 项 | 结果 | 证据 |
|---|---|---|
| 精度必测矩阵 | 待硬件复测 | 2026-11-09 历史 75 例不能替代当前验收 |
| 确定性和敏感性 | 待硬件复测 | 仅历史 kernel 哨兵通过 |
| 黄金回归 100% | CPU 通过 / 硬件待复测 | 当前 CPU 93/93；硬件 93/93 为历史 |
| 三层测试 | 待硬件复测 | 75 / 9 / 3 均为历史结果 |
| 性能与 dispatch | 待复测 | 历史产物不可独立复核，中央 NEW 不是门禁 |
| 报告与目录 | 完成 | REPORT、五份分册、catalog、生成索引已统一状态 |

## 6. 结论

表中 75/9/3、硬件 93/93、36/36 是 2026-11-09 修改前历史基线。2026-11-27 当前 PR 已通过 200 项 CPU 单测、证据检查和 CPU 黄金两路径各 93/93。分阶段探测中基础 copy、native sigmoid 及仓库 sigmoid Triton 的 257/255 最小用例通过；真实 C2 baseline 在 TP worker 的 `ncclCommInitRank` 阶段因 NCCL 错误退出。generic Triton 探针因自身 `tl` 作用域错误无效，不作为平台结论。修改后的三层、Triton 黄金、C2 和性能矩阵尚未完成。当前可以继续代码评审，但必须在最终合并前补齐硬件复测，且不声明普遍加速、其他平台支持或 backward。原始日志与归因见[运行时故障材料](../evidence/p800-runtime-failure-2026-11-27/ANALYSIS.md)。
