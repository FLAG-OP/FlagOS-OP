# aten.sigmoid 测试报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / Triton |
| 目标 / 日期 | 昆仑芯 P800 物理 2 号卡 / 2026-11-09 |
| 环境 | 官方 FlagGems 5.0 软件锁 |
| 结论 | P800 forward C1 范围通过；Must 项完成 |

## 1. 覆盖与结果

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel | 19 shape、3 优化 dtype、3 回退 dtype、布局、IEEE、哨兵 | PASS，75 例 |
| A1 op | 包态注册、counter、非连续/空张量和回退 | PASS，9 次命中 |
| C1 应用 | `2×64×4096` Transformer 激活，3 dtype | PASS，3/3 |
| 黄金 | 边界、长尾、Transformer、饱和值 | PASS，93/93 |
| 性能矩阵 | 4 shape × 3 dtype × 3 实现，20+100 | PASS，36/36 |
| 中央门禁 | `ops.sigmoid.triton` | OK：0 FAIL、1 NEW |
| dispatch | 直调与 A1 注册对照 | 完成：增量 `7.908 μs` |

## 2. 精度

kernel 层自研与参考逐位一致，max abs/rel 均为 0。黄金回归最差 abs `1.192e-07`、rel `1.680e-07`，低于 float32 `2e-6` 容差；float16/bfloat16 均逐位一致。NaN mask、空张量、确定性和输入敏感性通过。

## 3. 性能

正式 5.0 矩阵中自研/原生比值为 `1.158–4.276×`，当前没有组合快于原生。最接近原生的是 `2048×2048 bf16`；小尺寸主要受固定启动成本限制。性能结论不影响功能验收，但限制默认应用范围。

## 4. 环境注意

单卡容器必须同时设置 `XPU_VISIBLE_DEVICES=2` 和 `CUDA_VISIBLE_DEVICES=2`，使物理 XPU 2 映射为逻辑 `cuda:0`。标准 `p800-kunlunxin-gems5` profile 预期双卡布局，本次使用软件锁相同的本地单卡 profile。

## 5. 验收清单

| Must 项 | 结果 | 证据 |
|---|---|---|
| 精度必测矩阵 | PASS | 3 优化 dtype、19 shape、布局与特殊值 |
| 确定性和敏感性 | PASS | kernel 哨兵 |
| 黄金回归 100% | PASS | 93/93 |
| 三层测试 | PASS | 75 / 9 / 3 |
| 性能与 dispatch | PASS | 36/36；中央 0 FAIL；dispatch 已测 |
| 报告与目录 | PASS | REPORT、五份分册、catalog、生成索引 |

## 6. 结论

可以按 P800 forward C1 证据提交评审。当前不声明普遍加速、其他平台支持、backward 或 C2。
