# aten.sigmoid 开发报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / torch + Triton（P800 backend） |
| 目标设备 | 已提交 profile `p800-kunlunxin-gems5`；物理 XPU 1/2，算子层 `cuda:1`，vLLM TP=2 |
| 日期 / 状态 | 2026-11-27 / 当前 P800 非 C2 验收完成；C2 受 NCCL 初始化阻塞 |

## 1. 环境

使用仓库登记的 `p800-kunlunxin-gems5` 和官方 FlagGems 5.0
软件锁。容器映射物理 `/dev/xpu1`、`/dev/xpu2`，两个可见设备变量
均为 `1,2`；物理 XPU 1/2 按可见顺序对应逻辑 `cuda:0/1`。算子层
使用 `cuda:1`，框架层使用两卡 TP=2、`mp` executor。

| 组件 | 版本 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.9.0+cu129 |
| Triton | 3.0.0+f69a5406 |
| FlagGems | 5.0.0 |
| torch_xmlir | 1.0.0.1 |
| xtorch_ops | 0.1.2935+50a5d6a4 |
| XPU-RT / 设备 | 5.0.21 / P800 OAM 96 GiB |

## 2. 定义

接口为 `aten::sigmoid(Tensor self) -> Tensor`，逐元素计算 `1/(1+exp(-x))`。float16、bfloat16、float32 使用 fp32 中间精度并转回输入 dtype；int32、int64、bool 通过原生 redispatch 获得 float32 输出。测试覆盖 0D～4D、空张量、非连续布局、launch 边界、Transformer 激活和 IEEE/饱和值。

## 3. 实现

A1 在 profile 指定的 CUDA dispatch key 上注册。P800 后端使用 FlagGems `pointwise_dynamic` 和 `CodeGenConfig`；通用 Triton 后端保留尾块 mask 和大张量 block 65536。两者均在 `torch_device_fn.device` 上下文运行且不使用 autotune。不支持的 dtype/layout 使用 `CompositeExplicitAutograd` redispatch，避免注册递归。

2026-11-22 针对连续 `numel=255` 增加固定 256 lane 的专用 Triton kernel：所有 load/store 均无 mask，第 256 lane 的输入下标钳制到 254，结果存入 256 项底层存储后裁剪。该路径替代了会触发 255 元素设备 copy 的 padding 方案。

| 层级 | 状态 | 说明 |
|---|---|---|
| torch | 完成 | ATen 组合基线 |
| Triton | 完成 | 通用入口 + P800 pointwise backend |
| 硬件级 | 未实现 | XTC/xcc 不在本次范围 |

## 4. 验证状态

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel | 当前 PASS | 75 例、19 shape，max abs/rel=0，哨兵通过 |
| A1 op | 当前 PASS | 9 次命中、契约计数 9 次，布局与回退通过 |
| C1 应用 | 当前 PASS | `2×64×4096` 的 fp16/bf16/fp32 3/3 通过 |
| C2 vLLM | 环境初始化失败 | 2026-11-27 baseline TP worker 报 NCCL error，未进入生成/注入 |
| 黄金 | 当前 PASS | P800 Triton 93/93；最差 abs `1.192e-07` |

## 5. 性能

2026-11-27 当前口径为每组合独立子进程、warmup 20、计时 100 次并显式同步。当前 36/36 组合中，自研 12 个 shape/dtype 组合均慢于原生，比值范围 `1.198–5.736×`；原始 JSON 已入库。详见 [performance.md](performance.md)。

中央性能入口记录 `ops.sigmoid.triton=0.058 ms`，因没有历史 baseline 标记 NEW；结果为 0 FAIL，但不能据此宣称已有回归保护。

## 6. 风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 当前 12 个组合均未超过原生 | 不适合默认替换所有 sigmoid | 本次如实记录；后续路由或 kernel 优化 |
| 2 | 只在 P800 实机验证 | 不能外推其他五个平台 | 本次范围仅 P800，其他平台独立复测 |
| 3 | 当前 forward/C1 已完成，真实 vLLM C2 未完成 | backward 范围外；不能宣称 C2 通过 | C2 脚本已具备，通信环境恢复后采集 |
| 4 | 非连续输入会连续化 | 不保留原 stride | 数值与 shape 已验证 |
| 5 | TP=2 vLLM worker 在 `ncclCommInitRank` 阶段失败 | 只阻塞 C2；不影响已完成的单卡三层、黄金和性能证据 | 已归档故障材料；等待平台通信环境恢复 |

## 7. 结论与后续

2026-11-27 当前 PR 已通过 200 项 CPU 单测与证据检查，并完成 P800 三层、Triton 黄金和性能矩阵；可复核材料见[当前验收记录](../evidence/p800-current-validation-2026-11-27/ANALYSIS.md)。C2 仍受 TP=2 NCCL 初始化问题阻塞。后续在环境恢复后补 C2，并基于当前性能结果评估按 `numel/dtype` 的 native 路由或启动成本优化。

## 复现命令

```bash
export XPU_VISIBLE_DEVICES=1,2
export CUDA_VISIBLE_DEVICES=1,2
python ops/sigmoid/example.py p800-kunlunxin-gems5
python ops/sigmoid/script/check_accuracy.py --impl triton --device p800-kunlunxin-gems5
python ops/sigmoid/script/bench_matrix.py --profile p800-kunlunxin-gems5 --warmup 20 --iters 100
python ops/sigmoid/script/bench_dispatch.py --device p800-kunlunxin-gems5
python ops/sigmoid/script/c2_real_vllm.py --profile p800-kunlunxin-gems5
```
