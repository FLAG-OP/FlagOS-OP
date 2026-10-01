# 算子测试报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

| 项 | 值 |
|---|---|
| 算子名称 / 路线 / 级别 | `aten::_scaled_dot_product_attention_math` / **A1** / Triton（NPU）+ torch（P800/CPU） |
| 目标设备 / 测试日期 | `ascend910` · `p800-kunlunxin` · `cpu` / 2026-09-30~10-01 |
| 结论 | **通过**（Must 清单全勾：必测矩阵 ☑ · 哨兵 ☑ · 黄金 100% ☑ · 三层全绿 ☑ · perf 门禁 FAIL 0 ☑ · 四件套 ☑） |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ☑ | `test/kernel_level.py [ascend910\|cpu\|p800-kunlunxin]` | PASS：52 组（16 shape × 3 dtype + 全遮蔽/哨兵/dropout/错误路径）×3 profile |
| 框架层 | ☑ | `test/op_level.py [ascend910\|cpu]` | PASS：21 项（四模式拦截 · 注册=直调逐位 · 6 组 gradcheck · F.sdpa(MATH)） |
| 应用层 | ☑ | `test/framework_level.py [ascend910\|cpu]` | PASS：8 项（mini-decoder + 概率图消费者双跑） |
| 黄金回归 | ☑ | `script/gen_golden.py` + `check_accuracy.py` | **175/175** ×3 实现；原生对照 131/131（44 bool 跳过） |
| 性能门禁 | ☑ | `scripts/perf_run.py --pattern sdpa_math` + `perf_compare.py` | OK：FAIL 0 · WARN 0（triton 0.504ms / 8.524 TFLOPS 入库基线） |
| 语义证据 | ☑ | `probes/native_semantics.py [ascend910\|cpu]` | 6 节全过（schema / dropout 双规则表 / bool 怪癖 / 冲突边界 / F.sdpa 参数形态 / 注册键） |
| 跨层一致性 | ☑ | 黄金生成的三重互验（§3） | 参考 vs `F.sdpa(MATH)` vs 原生直调 vs 自洽式，全一致 |
| 一键 | ☑ | `python3 example.py [ascend910\|cpu\|p800-kunlunxin]` | 三层全绿 |

## 2. 环境

Ascend910_9382 · torch 2.10.0+cpu · torch_npu 2.10.0 · triton 3.5.1 ·
flag_gems 5.3.5 · python 3.11.15 · profile `configs/devices/ascend910.yaml`
（`scripts/check_env.py` 因缺 `configs/env/ascend910.lock.yaml` 不可用，
环境以 [reports/development.md](development.md) §1 表为准）。

## 3. 精度结果

### 3.1 三方（实为四实现）对照——黄金 175 组

容差：fp32 `1e-5` abs · fp16/bf16 `2e-2` abs · extreme `1e-3` rel。

| 实现 | 结果 | 最差 abs | 备注 |
|---|---|---|---|
| 自研 triton（NPU） | **175/175** | 3.906e-3（bf16） | [reports/accuracy.md](accuracy.md) |
| 自研 torch（CPU） | **175/175** | 1.953e-3（bf16） | |
| 参考 reference | **175/175** | 0 | 判卷标准本体 |
| 原生（未注册进程直调） | **131/131** | 9.766e-4（bf16，NPU） | 44 组 bool mask 按规则跳过（§5 #1） |
| FlagGems | 不适用 | — | 无"返回概率图"的同语义实现；其 SDPA 为融合算法、不产出 P（Should 项已注明原因） |

黄金生成时**四重互验**：①`F.sdpa(MATH)` ②原生直调（out+P）③
`out == (P/(1-p))@V` 自洽式（dropout 时）④参考实现，任一不一致即报错
——175 组全部通过后才落盘。

### 3.2 kernel 层直测（52 组/profile，16 shape × 3 dtype + 边界）

| dtype | NPU（triton）max\|Δout\| | CPU（torch）max\|Δout\| | max\|ΔP\|（两 profile） | 容差 | 判定 |
|---|---:|---:|---:|---:|---|
| fp32 | 1.79e-7 | 2.38e-7 | 1.22e-4 | 1e-5（out）/ 同档（P） | ✓ |
| fp16 | 1.95e-3 | 1.22e-4 | 1.22e-4 | 2e-2 | ✓ |
| bf16 | 7.81e-3 | 4.88e-4 | 1.22e-4 | 2e-2 | ✓ |

注：`ΔP` 最差 1.22e-4 出现在 bf16 概率图（量化级），远低于 2e-2。

**哨兵**：同输入两次逐位一致（确定性）；输入改动输出必变（敏感）；
随机 dropout 同种子 fwd/bwd 完全可复现。

**边界与错误路径**：全遮蔽行 `P=O=0` 无 NaN；`Sq≠Skv`（6×4/4×6）、
`decode Sq=1`、`Hkv=1` 广播、`D=80` 非 2 幂、`S=100/197` 非整倍数；
`causal+mask`（bool/float 皆然）与 `GQA 头不整除` 均按 native 报错。

**dropout 统计判定**：显式掩码下与参考逐位一致（fp32 1e-5 内）且满足
native 怪癖 `O = (P/(1-p))@V`；随机路径 keep 率 0.499（期望 0.5）、
kept 上缩放 2.000（期望 `1/(1-p)`）。

### 3.3 框架层（21 项）

| 检查 | 结果 |
|---|---|
| 拦截计数：grad 开启 / no_grad / **inference_mode** / 无 requires_grad | 4/4 命中（+1 各）；inference_mode 为成对注册的回归项 |
| 注册路径 vs 直接调 impl | **逐位相等** |
| 注册路径 vs 原生基线 | NPU 3.58e-7 / CPU 4.17e-7 |
| gradcheck（fp64，6 组） | base / causal / float-mask ±grad / gqa / explicit-dropout / **概率图消费者 dP** 全过 |
| out-消费者 / P-消费者梯度 vs 参考反传 | 4.3e-7 / 7.5e-8（NPU） |
| `F.sdpa(MATH)` 拦截 + 输出=参考 | CPU 键命中，err 2.98e-7 |
| 错误路径 | causal+mask、GQA 不整除均抛出 native 同款异常 |
| 拦截总次数 | 3822（gradcheck 逐点调用计入） |

### 3.4 应用层（8 项）

| 指标 | NPU（triton） | CPU（torch） |
|---|---:|---:|
| 消费方命中次数 | 21（前向 3 + 每次反向若干） | 21 |
| logits L∞ vs 原生 | 1.86e-8 | 0 |
| top-1 一致率 | 1.000 | 1.000 |
| 概率图 L∞ | 2.98e-8 | 0 |
| 全权重梯度 L∞（含 dprobs） | 1.75e-10 | 5.82e-11 |
| 贪心序列逐步一致率 | 1.000（8/8） | 1.000（8/8） |
| inference_mode 消费方 | 命中 +2 | 命中 +2 |

## 4. 性能结果

| 实现 | NPU fp16 1k D128 | 基线对比 | 精度代价 |
|---|---:|---|---|
| 自研 triton | 0.515-0.561ms | 基线 | 0（同过 175/175） |
| 自研 A1 包装 | 0.572ms | +0.034ms（autograd.Function） | 0 |
| 原生 math（同返回 out+P） | 0.695ms | 1.29x 慢于自研 | 0（131/131） |
| 原生 F.sdpa（融合，**不返回 P**） | 参考值见 [performance.md](performance.md) | 输出契约不同，不可直接判卷 | 不适用 |

门禁：`perf_compare --device ascend910` → **OK，FAIL 0 · WARN 0**
（triton 0.543ms vs 基线 0.504ms，+7.8%，阈值内）。
本算子延迟 >100µs，不触发微算子 `bench_dispatch` 附加项。

### 2026-10-01 P800 复验

| 层级 | 结果 |
|---|---|
| kernel | 52/52；`torch_level` direct 与 CPU reference 全部对齐 |
| 黄金 | `--impl torch --device cuda:1` **175/175**，worst 9.766e-4 |
| A1 | `AutogradCUDA+CUDA` 成对注册；21/21 项通过（含 6 组 fp64 gradcheck） |
| 应用层 | `flag_gems.only_enable(['gelu'])` 后 9/9；logits / 概率图 / 梯度 diff 均为 0 |
| perf gate | `ops.sdpa_math.p800` / `.native` 入库；FAIL 0 · WARN 0 |

P800 direct 性能 **1.03-1.27x** native；A1 生产相关大 shape
**1.02-1.24x**。复现：

```bash
python3 example.py p800-kunlunxin
python3 script/check_accuracy.py --impl torch --device cuda:1
python3 script/bench_perf.py --device cuda:1 --register \
  --json-out reports/perf_p800-kunlunxin.json
```

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | `bool attn_mask` 直调的 0/1 加性怪癖 vs 本实现的 `-inf` 遮蔽 | 44 组黄金在 `--impl native` 下跳过；`F.sdpa` 调用方不受影响 | 有意分歧（[development.md](development.md) §6 #1） |
| 2 | 只注册 `Autograd*` 时 inference_mode 不命中 | 推理路径绕过自研实现 | 已修（成对注册），有前后对照证据 |
| 3 | decode/tail 直调 0.92-0.97x 慢于原生 | 小形状启动占比高 | **已优化**：单 kernel 融合后 1.16-1.17x（[performance.md](performance.md) §6）；A1 包装路径仍 0.86x |
| 4 | `configs/env/ascend910.lock.yaml` 缺失 | `check_env` 无法核对 | 记录在 development §1；黄金/性能数字均注明环境 |
| 5 | `torch.library` 注册不可撤销 | 同进程测试必须先采原生基线 | bench/test/probe 均已按"先基线后注册"实现 |

## 6. 结论

对照 [Must 清单](../../../docs/acceptance.md#levels)：
精度必测矩阵 ☑（非整倍数 shape、三 dtype、特殊用例、≥2 seed/组合）·
哨兵 ☑ · 黄金 100% ☑ · 三层全绿 ☑ · perf 门禁 0 FAIL ☑ ·
报告四件套 ☑ → **可交付**。Should 项（三方对比中 FlagGems 缺席）已给出
原因：无同语义（返回概率图）实现。
