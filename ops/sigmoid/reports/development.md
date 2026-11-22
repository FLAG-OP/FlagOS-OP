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

2026-11-22 针对连续 `numel=255` 增加固定 256 lane 的专用 Triton kernel：所有 load/store 均无 mask，第 256 lane 的输入下标钳制到 254，结果存入 256 项底层存储后裁剪。该路径替代了会触发 255 元素设备 copy 的 padding 方案。

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
| 3 | 当前只完成历史 forward/C1 基线 | backward 范围外；真实 vLLM C2 当前待复测 | C2 脚本已具备，运行时恢复后采集 |
| 4 | 非连续输入会连续化 | 不保留原 stride | 数值与 shape 已验证 |
| 5 | 当前服务器普通 `1×14336` Triton 也会超时 | 无法完成修改后硬件全量复测 | 静态/CPU/代表点先完成；等待平台运行时恢复 |

## 7. 结论与后续

2026-11-09 历史基线满足 C1 与性能评审条件；2026-11-22 当前代码的静态、CPU 和 255 代表点验证已完成，但修改后三层、Triton 黄金、C2 和性能矩阵待运行时恢复。后续优先完成这些最终复测，再评估按 `numel/dtype` 的路由或启动成本优化。

## 复现命令

```bash
python ops/sigmoid/example.py p800-liyanz-local
python ops/sigmoid/script/check_accuracy.py --impl triton --device p800-liyanz-local
python ops/sigmoid/script/bench_matrix.py --profile p800-liyanz-local --warmup 20 --iters 100
python ops/sigmoid/script/bench_dispatch.py --device p800-liyanz-local
```
