# 子图融合 vs 截流下发实验（2026-09-22 实测）

> 问题: Triton 层级差距太大（12x），能否用子图融合救？
> 还是实际使用时应从 FlagGems 截流下发给厂商算子？

## 实验A: torch.compile/子图融合——**救不了，且本栈当前不可用**

| 路径 | 结果 |
|---|---|
| eager ours (Triton) | 1.434 ms |
| eager native | 0.224 ms |
| compile ours | **失败**: inductor 与 triton-ascend 3.5 不兼容（`cannot import triton_key`） |
| compile native | 同上 |

两个层面都不成立:
1. **机制上**: SDPA 本身已是融合算子（softmax+2×GEMM 单 kernel），
   差距在融合 kernel **内部**的 GEMM codegen（纯 matmul 就慢 5.2x，
   见 perf_analysis）。子图融合优化的是**算子间**开销（launch 间隙/
   中间张量往返），而我们的 launch 不是瓶颈（warps/stages 零敏感）。
   融合救不了"融合算子的内部效率"
2. **工程上**: torch 2.10 + torch_npu 的 inductor 后端连不上
   triton-ascend 3.5（API 断裂）——本栈 compile 路径当前直接不可用

## 实验B: 截流下发（shape-aware 混合路由）——**可行且收益立现**

原型（`script/exp_fusion_vs_dispatch.py` 的 `sdpa_smart`）:
S≥1024 且无 mask → 原生直通；其余（小 S / 带自定义 mask / native
不支持的组合）→ 自研 Triton:

| S | ours | native | smart 路由 | smart/最优 |
|---|---|---|---|---|
| 256 | 0.107 | 0.073 | 0.110 (→ours) | 1.51x |
| 512 | 0.128 | 0.095 | 0.128 (→ours) | 1.35x |
| 1024 | 0.399 | 0.124 | 0.128 (→native) | **1.03x** |
| 2048 | 1.439 | 0.228 | 0.225 (→native) | **0.99x** |
| 4096 | 5.526 | 0.522 | 0.519 (→native) | **0.98x** |
| 8192 | 21.509 | 1.835 | 1.799 (→native) | **0.98x** |

大 S 段拿到 **98-99% 最优**（比纯 ours 快 12x）；小 S 段 1.3-1.5x
开销来自路由判断本身可忽略（0.003ms 级）+ 小 S 时 ours 本就接近
native（1.4x）。若路由阈值降到 S≥256 则全段 ≤1.05x。

## 结论与建议

1. **"从 FlagGems 截流下发厂商算子"就是 FlagOS 体系的标准答案**——
   这正是 B 路线（vendor:ascend 身份注册 OpManager，与 default.flagos
   同网竞争）+ `VLLM_FL_PER_OP` 钉选的设计意图。我们的混合路由原型
   就是它的最小形态
2. **正确的生态分工**（呼应 development.md §8）:
   - 峰值场景（大 S prefill）: 厂商 kernel（CANN 闪电注意力，61% 峰值）
   - 长尾/自定义场景: FlagGems/Triton（可移植、可改，如 4D mask/
     GQA 组合、native 不覆盖的 shape）
   - 路由层: shape-aware 策略或 vllm_fl 的 policy 钉选
3. 本实现的价值定位不变: 它是"长尾兜底 + 验证基准 + 栈演进受益者"
   ——BiShengIR GEMM 效率每提升一档，路由阈值自动下移

## 复现

```bash
python3 script/exp_fusion_vs_dispatch.py
```
