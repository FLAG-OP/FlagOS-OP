# aten.sigmoid 开发报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

| 项 | 值 |
|---|---|
| 路线 / 级别 | A1 / torch + Triton（P800 backend） |
| 目标设备 | P800 物理 XPU 2；单卡 profile `p800-liyanz-local` |
| 日期 / 状态 | 2026-11-09 / 实现与自测完成（C1） |

## 1. 环境

使用仓库登记的官方 FlagGems 5.0 镜像，物理 2 号卡独占映射为容器 `cuda:0`。本地 profile 只改变可见设备与逻辑编号，软件包与 `p800-kunlunxin-gems5` 锁一致，`scripts/check_env.py` 全部通过。

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

| 层级 | 状态 | 说明 |
|---|---|---|
| torch | 完成 | ATen 组合基线 |
| Triton | 完成 | 通用入口 + P800 pointwise backend |
| 硬件级 | 未实现 | XTC/xcc 不在本次范围 |

## 4. 验证

| 层级 | 结果 | 明细 |
|---|---|---|
| kernel | PASS | 75 例、19 shape、误差 0、确定性和敏感性哨兵通过 |
| A1 op | PASS | 9 次命中，counter 同为 9，三优化/三回退 dtype |
| C1 应用 | PASS | `2×64×4096` Transformer 激活，3/3 dtype 输出一致 |
| 黄金 | PASS | 93/93；最差 abs `1.192e-07`、rel `1.680e-07` |

## 5. 性能

正式口径为每组合独立子进程、warmup 20、计时 100 次并显式同步。36/36 组合完成；自研 12 个 shape/dtype 组合均慢于原生，比值范围 `1.158–4.276×`。大张量 bf16 最接近原生，小尺寸受约 49～58 μs 固定成本影响。A1 直调 `47.970 μs`、注册 `55.877 μs`，增量 `7.908 μs`（16.5%）。详见 [performance.md](performance.md)。

中央性能入口记录 `ops.sigmoid.triton=0.058 ms`，因没有历史 baseline 标记 NEW；结果为 0 FAIL，但不能据此宣称已有回归保护。

## 6. 风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 当前 12 个组合均未超过原生 | 不适合默认替换所有 sigmoid | 本次如实记录；后续路由或 kernel 优化 |
| 2 | 只在 P800 实机验证 | 不能外推其他五个平台 | 本次范围仅 P800，其他平台独立复测 |
| 3 | 只覆盖 forward/C1 | 不含 backward 和真实模型引擎 | 明确列为范围外 |
| 4 | 非连续输入会连续化 | 不保留原 stride | 数值与 shape 已验证 |

## 7. 结论与后续

P800 forward 的功能、精度、注册、C1 和性能基线满足提交评审条件。后续优先验证按 `numel/dtype` 的安全原生路由，或降低 P800 pointwise 启动成本；第二个平台出现时再增加 `PLATFORM.md`/`MERGE.md` 和独立证据。

## 复现命令

```bash
python ops/sigmoid/example.py p800-liyanz-local
python ops/sigmoid/script/check_accuracy.py --impl triton --device p800-liyanz-local
python ops/sigmoid/script/bench_matrix.py --profile p800-liyanz-local --warmup 20 --iters 100
python ops/sigmoid/script/bench_dispatch.py --device p800-liyanz-local
```
