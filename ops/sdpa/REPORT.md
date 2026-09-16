# 算子总体报告: sdpa（scaled_dot_product_attention）

| 项 | 值 |
|---|---|
| 算子名称 | `aten::scaled_dot_product_attention`（SDPA / flash-attention 族） |
| 语义 | [reference.py](reference.py)（softmax(QKᵀ·scale + mask)·V，fp32 内部计算） |
| 目标设备 / 路线 | `ascend910`（Ascend 910, CANN 9.0.0）/ **A1**（aten 拦截） |
| 平台绑定 | 本交付为 ascend910 专属——绑定项与移植预案见 [PLATFORM.md](PLATFORM.md) |
| 多平台合并 | 第二平台实现完成后按 [MERGE.md](MERGE.md) 七步合入（黄金/测试/注册链复用） |
| 日期 / 状态 | 2026-09-16 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | ATen 组合（第二判卷人/对照） |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ | **主实现**: online-softmax two-pass，causal 截断，GQA/双 mask/尾块 |
| 硬件级 | — | ⬜ 置空 | 原生 CANN 闪电注意力即该级参照（见性能分册差距分析） |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| kernel 层 | `test/kernel_level.py`（36 组精度 + 哨兵） | ✅ 37/37 | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py`（A1 拦截 + 梯度） | ✅ | — |
| 应用层 | — | ⬜ | 本栈 vllm 0.20.2+empty 为空壳，无真实推理可注入（[说明](README.md#应用层)） |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 265/265 | [goldendata/](goldendata/) |
| 性能 | `script/bench_perf.py` 三方对照 | ✅ | [reports/performance.md](reports/performance.md) |
| 平台守卫 | `probes/guard_check.py`（元数据/调用/注册三层） | ✅ | [PLATFORM.md](PLATFORM.md) §5 |

## 关键数字（fp16，S=2k D=128 H=16 causal）

| 指标 | 自研 Triton | FlagGems 5.3.5 | 原生 CANN |
|---|---|---|---|
| 精度（vs CPU fp32 黄金） | ✅ 265/265 | 未接入 aten 分发 | ✅ 265/265 |
| prefill_2k 延迟 | 1.44 ms | 25.0 ms | **0.22 ms** |
| 相对 | 1.00x | **0.06x（慢 17.4x）** | 6.5x 快 |

## 结论与遗留

两层验证 + 黄金全绿；**比 FlagGems 现有 Triton SDPA 快 3.3-17.6x**，
落后原生 CANN（AscendC 手写闪电注意力）3.3-10.8x——差距根因为
Triton→BiShengIR 栈的 GEMM 效率与 UB 约束下的 tile 上限，非 kernel
结构问题（[性能分册](reports/performance.md)含纯 matmul 天花板证据）。
遗留: ①应用层空缺（本栈无可用 vllm）②fp32 ieee 精度模式性能代价
未单独优化 ③PyPTO 重写路线需 910B+ 环境（见开发报告§7）。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告（7 章） | [reports/development.md](reports/development.md) |
| 测试报告（范围矩阵） | [reports/test-report.md](reports/test-report.md) |
| 精度分册（含复现命令） | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册（含根因分析与复现） | [reports/performance.md](reports/performance.md) |
| 性能根因专项（变体实验） | [reports/perf_analysis.md](reports/perf_analysis.md) |
| 平台绑定清单与移植指引 | [PLATFORM.md](PLATFORM.md) |
| 多平台合并指南 | [MERGE.md](MERGE.md) |
| 开发证据链（探针归档） | [probes/](probes/README.md) |
