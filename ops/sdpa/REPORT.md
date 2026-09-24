# 算子总体报告: sdpa（scaled_dot_product_attention）

| 项 | 值 |
|---|---|
| 算子 | `aten::scaled_dot_product_attention`（SDPA / flash-attention 族） |
| 语义 | [reference.py](reference.py)（fp32 内部、GQA、causal、bool/float mask） |
| 路线 | A1 aten 拦截；平台实现由 [kernel/triton_level.py](kernel/triton_level.py) facade 分发 |
| 平台 | ascend910（Triton 主实现）· p800-kunlunxin（厂商委托 + fp32 补偿）· mlu590（TMO FA + fused overrideable 委托） |
| 状态 | 2026-09-23: MLU590 三层 + 黄金全绿（fused/TMO 修订后 1.00-1.33x 原生）；2026-09-22: P800 三层 + 黄金全绿 |

## 实现矩阵

| 平台 | 级别 | 文件 | 状态 | 备注 |
|---|---|---|---|---|
| ascend910 | Triton | [kernel/backends/ascend910.py](kernel/backends/ascend910.py) | ✅ | online-softmax two-pass、causal 截断、GQA/双 mask/尾块 |
| p800-kunlunxin | 厂商 kernel 委托 | [kernel/backends/p800_kunlunxin.py](kernel/backends/p800_kunlunxin.py) | ✅ | fp16/bf16 直调 efficient attention；fp32 走 ATen 组合 + bmm workaround |
| mlu590 | 厂商 kernel 委托 | [kernel/backends/mlu590.py](kernel/backends/mlu590.py) | ✅ | TMO FA（半精度）→ fused overrideable；math/FlagGems 兜底 |
| 通用 | torch | [kernel/torch_level.py](kernel/torch_level.py) | ✅ | 第二判卷人 / 对照 |
| 硬件级 | — | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 | P800 硬件 SDK 不在当前容器；XMLIR/Triton 试验仍显著慢于厂商 kernel |

## mlu590 验证矩阵

| 层级 | 命令 / 入口 | 结果 |
|---|---|---|
| kernel | `python3 test/kernel_level.py mlu590` | ✅ 37/37，最大误差 3.91e-3（bf16），哨兵通过，1k D128 0.25ms |
| 黄金 | `python3 script/check_accuracy.py --impl triton --device mlu:0` | ✅ 265/265，worst 7.81e-3（bf16 bool，容差内） |
| op | `python3 test/op_level.py mlu590` | ✅ AutogradPrivateUse1 拦截、注册=直调(逐位)、vs 原生 4.88e-4、梯度通过 |
| 应用 | `python3 test/framework_level.py mlu590` | ✅ mini-decoder 拦截 28 次，logits diff=1.95e-3，续写一致率 1.00 |
| 守卫 | `python3 probes/guard_check.py mlu590` | ✅ 元数据 / mlu 路径 / CPU 拒绝 / 注册 |

复现报告与坑位：[reports/mlu590.md](reports/mlu590.md)。

## mlu590 fp16 性能采样

数据：[perf_fp16_mlu590.json](reports/perf_fp16_mlu590.json)
（warmup=20，iters=100；speedup = native/ours，**>1 表示 ours 更快**）。

| shape | ours | Python F.sdpa | FlagGems | speedup=F/ours |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.227ms | 0.301ms | 2.915ms | **1.33x** |
| prefill 1k D128 | 0.283ms | 0.363ms | 3.666ms | **1.28x** |
| prefill 2k D128 | 0.351ms | 0.445ms | 8.088ms | **1.27x** |
| prefill 4k D128 | 0.727ms | 0.902ms | 26.628ms | **1.24x** |
| GQA 1k D128 | 0.366ms | 0.442ms | 5.176ms | **1.21x** |
| decode D128 | 0.225ms | 0.224ms | 1.989ms | **1.00x** |

路径：半精度走 TMO FA（失败则 fused overrideable）；fp32 走
overrideable。与原生常逐位/亚 ulp 一致；FlagGems Triton 慢 10-40x。

## p800-kunlunxin 验证矩阵

| 层级 | 命令 / 入口 | 结果 |
|---|---|---|
| kernel | `python3 test/kernel_level.py p800-kunlunxin` | ✅ 37/37，最大误差 3.906e-3（bf16），哨兵通过 |
| 黄金 | `python3 script/check_accuracy.py --impl triton --device cuda:1` | ✅ 397/397，extreme 相对误差 4.25e-3 |
| op | `python3 test/op_level.py p800-kunlunxin` | ✅ AutogradCUDA 拦截、注册=直调、direct 与 A1 梯度通过 |
| 应用 | `python3 test/framework_level.py p800-kunlunxin` | ✅ mini-decoder 拦截 28 次，logits diff=0，续写一致率 1.00 |
| 守卫 | `python3 probes/guard_check.py p800-kunlunxin` | ✅ 元数据 / CUDA 路径 / CPU 拒绝 / 注册 |
| perf gate | `scripts/perf_run.py --device p800-kunlunxin --pattern ops.sdpa` | ✅ P800/native 两条用例入库，FAIL 0 · WARN 0 |

复现报告与坑位：[reports/p800-kunlunxin.md](reports/p800-kunlunxin.md)。

## 关键数字

数据：[perf_fp16_p800-kunlunxin.json](reports/perf_fp16_p800-kunlunxin.json)
（warmup=20，iters=100；加速比 = baseline 延时 / ours 延时，>1 更快）。

| shape | ours | Python F.sdpa | FlagGems | 加速比 = F.sdpa/ours |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.1484ms | 0.1655ms | 0.3120ms | **1.115x** |
| prefill 1k D128 | 0.1442ms | 0.1574ms | 0.3606ms | **1.092x** |
| prefill 2k D128 | 0.2584ms | 0.2941ms | 1.3283ms | **1.138x** |
| prefill 4k D128 | 0.6632ms | 0.6796ms | 4.3990ms | **1.025x** |
| GQA 1k D128 | 0.1884ms | 0.2045ms | 0.5667ms | **1.085x** |
| decode D128 | 0.1132ms | 0.1270ms | 0.1611ms | **1.122x** |

### FA2 协议下的 A100 论文参照

同 `batch×S=16k`、causal、fp16 协议；A100 为 FlashAttention-2 论文图表
读数换算，非同机复测。P800 每次读取一个输出元素强制执行完成。

| 场景 | P800 ours | A100·FA2 | P800/A100 延时 |
|---|---:|---:|---:|
| D64 S=2k | 2.156ms | ~0.785ms | **2.75x 慢** |
| D64 S=8k | 7.040ms | ~2.894ms | **2.43x 慢** |
| D128 S=1k | 0.789ms | ~0.344ms | **2.30x 慢** |
| D128 S=4k | 2.312ms | ~1.195ms | **1.94x 慢** |
| D128 S=8k | 4.404ms | ~2.340ms | **1.88x 慢** |

结论：在当前栈上，直接选择底层 efficient attention 是合理交付；
自写 Triton 需先解决 XMLIR Triton 编译/launch 稳定性，不应为了
“实现语言必须是 Triton”放弃厂商成熟 kernel。

## 结论与遗留

Ascend 三层验证、黄金 265/265、性能对标与根因分析保持有效，详见
[reports/accuracy.md](reports/accuracy.md)、
[reports/performance.md](reports/performance.md)、
[reports/perf_analysis.md](reports/perf_analysis.md)。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 平台绑定清单 | [PLATFORM.md](PLATFORM.md) |
| 多平台合并指南 | [MERGE.md](MERGE.md) |
| P800 复现报告 | [reports/p800-kunlunxin.md](reports/p800-kunlunxin.md) |
| P800 direct Triton 探针 | [reports/p800-triton-probe.md](reports/p800-triton-probe.md) |
| P800 自研固定调度实验 | [reports/p800-custom-schedule.md](reports/p800-custom-schedule.md) |
| MLU590 复现报告 | [reports/mlu590.md](reports/mlu590.md) |
| MLU590 测试报告（已填样例） | [reports/examples/sdpa_mlu590_test_report.md](../../reports/examples/sdpa_mlu590_test_report.md) |
| MLU590 开发报告 | [reports/examples/sdpa_mlu590_report.md](../../reports/examples/sdpa_mlu590_report.md) |
| Ascend 开发报告 | [reports/development.md](reports/development.md) |
| Ascend 测试报告 | [reports/test-report.md](reports/test-report.md) |
| Ascend 精度 / 性能分册 | [reports/accuracy.md](reports/accuracy.md) / [reports/performance.md](reports/performance.md) |
| 性能 JSON | [reports/perf_fp16_p800-kunlunxin.json](reports/perf_fp16_p800-kunlunxin.json) · [reports/perf_fp16_mlu590.json](reports/perf_fp16_mlu590.json) |
