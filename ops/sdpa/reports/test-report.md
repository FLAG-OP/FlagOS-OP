# 算子测试报告: sdpa

| 项 | 值 |
|---|---|
| 算子名称 | `aten::scaled_dot_product_attention` |
| 实现路线 / 开发级别 | A1 torch 算子替换 / Triton（Ascend）+ 厂商委托（P800 · MLU590） |
| 目标设备 | `ascend910` · `p800-kunlunxin` · `mlu590` |
| 测试日期 | 2026-09-16（Ascend）· 2026-09-22（P800）· 2026-09-23（MLU590） |
| 结论 | **三平台通过**（MLU590 见 §7 与 [reports/mlu590.md](mlu590.md)；示例填充见 [sdpa_mlu590_test_report](../../../reports/examples/sdpa_mlu590_test_report.md)） |

## 1. 测试范围

| 层级 | 是否覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ✅ | `test/kernel_level.py` | PASS（37 组 + 哨兵 + 快速性能；Ascend 36 组历史保留） |
| 框架层（注册/拦截/梯度） | ✅ | `test/op_level.py` | PASS（拦截命中 · 逐位一致 · 梯度 vs 原生） |
| 应用层（消费方双跑） | ✅ | `test/framework_level.py` | PASS（mini-decoder: 拦截 28 · logits 1.95e-3 · 续写一致率 1.00） |
| 黄金回归 | ✅ | `script/gen_golden.py` + `check_accuracy.py` | PASS（265/265，双向互验生成） |
| 性能三方对照 | ✅ | `script/bench_perf.py` | OK（fp16/bf16 各 6 形状） |
| 平台守卫（三层） | ✅ | `probes/guard_check.py` | PASS（元数据/调用拦截/注册正向） |
| 写法变体回归 | ✅ | `script/perf_variants.py` | PASS（8 变体数值全过；V4 已合入） |
| 一键复跑 | ✅ | `python3 example.py` | 三层全绿 |

## 2. 环境

Ascend 910_9382 (CANN 9.0.0) · torch 2.10.0+cpu · torch_npu 2.10.0 ·
triton 3.5.1 · flag_gems 5.3.5 · python 3.11.15（详见开发报告§1）。

## 3. 精度结果

容差: fp32=1e-5 abs · fp16/bf16=2e-2 abs · extreme 相对 5e-3。

### kernel 层直测（36 组 = 12 shape × 3 dtype）

| dtype | max err | 容差 | 判定 |
|---|---|---|---|
| fp32 | 1.2e-7 | 1e-5 | ✓（ieee dot） |
| fp16 | 9.8e-4 | 2e-2 | ✓ |
| bf16 | 3.9e-3 | 2e-2 | ✓ |

shape 覆盖: 基础/批量/尾块 100·333/ViT D=80/GQA/bool mask/float
mask/Sq=1 decode。哨兵: 确定性 ✓（两次调用位级一致）· 输入敏感 ✓。

### 黄金回归（265 组）

| 实现 | PASS | worst |
|---|---|---|
| 自研 Triton | 265/265 | 7.8e-3（bf16 bool-mask 组） |
| 原生 NPU（对照 shim） | 265/265 | 3.9e-3 |
| reference @ CPU | 265/265 | 0 |

### 框架层（A1 注册后）

拦截命中（count>0）；注册路径输出与直调**逐位一致**；vs 未注册
原生输出 1.2e-4（两实现数学等价的数值差）。梯度 vs 原生独立进程:
dq 4.9e-4 · dk 4.9e-4 · dv 3.9e-3。

## 4. 性能结果

@ S=2048 D=128 H=16 causal（详见[性能分册](performance.md)）:

| 实现 | 延迟 | 相对自研 |
|---|---|---|
| 自研 Triton | 1.44 ms | 1.00x |
| 原生 CANN | 0.22 ms | 6.5x 快 |
| FlagGems 5.3.5 直调 | 25.0 ms | 17.4x 慢 |

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 应用层为轻量消费方（非真实 vLLM） | 真实引擎行为待环境具备后补验 | 已按 softmax-fullstack 先例落地双跑 |
| 2 | 注册进程内不可撤销 | 混跑需子进程 | 已在测试中隔离 |
| 3 | 共享服务器性能扰动 | 绝对值 ±15µs | 用相对比值 + 空闲时段 |

## 6. 结论

### 2026-09-22 p800-kunlunxin 复验

| 层级 | 结果 |
|---|---|
| kernel | 37/37；普通 fp32 ≤5e-7，bf16 最大 3.906e-3；哨兵通过 |
| 黄金 | 265/265；extreme 相对误差 4.25e-3 |
| A1 op | `AutogradCUDA` 拦截命中；注册=直调；direct q/k/v 与可微 mask 反向通过；dq/dk/dv 梯度对照通过 |
| 应用 | mini-decoder 拦截 28；logits diff=0；贪心续写一致率 1.00 |
| 守卫 | 元数据、`cuda:1` 正向路径、CPU 拒绝、注册全通过 |

详细复现：[p800-kunlunxin.md](p800-kunlunxin.md)。

### 2026-09-23 mlu590 复验

| 层级 | 结果 |
|---|---|
| kernel | 37/37；fp32 1.79e-7 / fp16 4.88e-4 / bf16 3.91e-3；哨兵通过；1k D128 0.25ms |
| 黄金 | 265/265；worst 7.81e-3（bf16 bool，容差内） |
| A1 op | `AutogradPrivateUse1` 拦截；注册=直调逐位；vs 原生 4.88e-4；梯度 ≤6.1e-5 |
| 应用 | mini-decoder 拦截 28；logits 1.95e-3；贪心续写一致率 1.00 |
| 守卫 | 元数据、`mlu:0` 正向路径、CPU 拒绝、注册全通过 |
| 性能 | fp16 相对原生 **1.00–1.33x**（TMO→fused overrideable） |

详细复现：[mlu590.md](mlu590.md) ·
已填样例 [sdpa_mlu590_test_report](../../../reports/examples/sdpa_mlu590_test_report.md)。

### 总结论

kernel 层/框架层/应用层/黄金/性能五线全绿（ascend910 · p800-kunlunxin ·
mlu590 三平台），正确性与拦截机制均有实验证据链；性能定位清晰。
**建议合入**（真实 vLLM 场景待环境具备后补验；MLU 侧 quirks 已声明）。
