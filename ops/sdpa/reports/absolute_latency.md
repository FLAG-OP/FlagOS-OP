# 绝对延迟视角: 910 vs A100（FA2 协议同 FLOPs 换算）

> 前序报告都以"相对比值"呈现（利用率/加速比）。本文换**绝对延迟**
> 视角回答"典型场景能不能跟 A100 差不多"。
> A100 列为同 FLOPs 按论文 TFLOPS 换算值（±10% 粒度，非同机实测）；
> 910 列为本机实测（auto 路由最优侧 = CANN 原生）。

## 前向（fp16 causal，batch=16k/S 恒定 tokens，16 头 D=128）

| 场景 | 910 实测 | A100·FA2 | 比值 |
|---|---|---|---|
| prefill S=1k | 0.68 ms | 0.34 ms | 2.0x 慢 |
| prefill S=2k | 1.12 ms | 0.62 ms | 1.8x 慢 |
| prefill S=4k | 1.84 ms | 1.20 ms | 1.5x 慢 |
| prefill S=8k | 3.51 ms | 2.34 ms | 1.5x 慢 |

D=64 档差距略大（1.9-2.1x，A100 FA2 在小 D 上并行更满）。

## 结论: 接近但不到"差不多"——**1.5-2x 慢**是当前真实水位

- **能跟的程度**: 单看毫秒量级是同一数量级——S=4k prefill 差 0.6ms。
  在端到端 LLM 推理里，attention 只占一部分（还有 GEMM/归约/通信），
  1.5-2x 的 attention 差距摊到端到端通常缩到百分之十几
- **跟不上的部分**: 根源是硬件峰值（CANN 手写已到 61% 利用率，
  与 A100·FA2 的 75% 同档）——910 官方 fp16 峰值 256 TFLOPS vs
  A100 312 TFLOPS，理论比就是 1.22x，叠加利用率差（61% vs 75%）
  得 1.5x 左右，D=64 档因并行度问题再放大到 2x
- **纯 Triton 路径**（未经截流路由）则是 10-20x 慢——但生产配置
  已由 auto_dispatch 保证大 S 走原生

## 何时能"差不多"

1. **decode 场景**（S 小、GEMV-bound）: 受 memory bandwidth 主导，
   910 HBM ~1.2TB/s vs A100 ~2TB/s，理论比 1.7x——但 decode 的
   attention 只占端到端极小部分，实际感知接近
2. **短序列/小 batch**: 绝对值都在亚毫秒级，差异被 launch 开销
   掩盖
3. **等 PyPTO/CANN 迭代**: 利用率 61%→75% 一旦追平，绝对差距
   收敛到硬件峰值比 1.22x

## 复现

```bash
python3 script/perf_fa2_protocol.py    # 910 侧实测
# A100 列换算: flops / 论文 TFLOPS（reports/perf_a100.md 图8 数据）
```
