# KernelGen 官方 MCP 生成版 vs 手写版 vs 本地 LLM-Track 对比（终版）

日期: 2026-09-17 · 通道: kernelgen.flagos.io/sse（streamable HTTP）,
generate_kernel 官方工具 · 参数按 flagos-skills 协议（arg_* 为逗号串）

## 三方终态对比（同一四阶段验证协议）

| 维度 | 手写版 | KernelGen 官方版 | 本地 LLM-Track |
|---|---|---|---|
| 编译 | ✅ | ⚠️ 需人工 patch ×1（if/else 运行时赋值，ascend codegen 不支持） | ✅ |
| 正确性 | 36/36 | **7/9**（patch 后；mask 仅 2D 语义，4D bool/float mask 错位） | 3/5（尾块越界 + causal 硬编码） |
| 哨兵 | ✅ | ✅ | ✅ |
| 性能 S=2k | **1.437ms (6.2x nat)** | 1.523ms (6.6x nat, 1.06x 手写) | 2.935ms (13.3x nat) |
| 平台健壮性 | 内置（device ctx/同 dtype dot/ieee/无 exp2） | 无（裸生成，CUDA 方言假设） | 无 |

## 官方版细节发现

1. **生成质量显著高于本地 LLM-Track**: 尾块 masked load / causal 截断 /
   dropout / 尾块小 S 自适应 BLOCK——flagos_wiki 提示被有效消费
   （对比本地版连尾块都漏）
2. **但写法是 CUDA 方言**: `if IS_CAUSAL: end_n=...` 运行时赋值在
   triton-ascend 直接编译失败（UnsupportedLanguageConstruct）——
   **跨芯片方言差异是生成代码的第一道坎**
3. **patch 后暴露真实缺陷**: 4D attn_mask 只取 stride(-2)/stride(-1)，
   batch/head 维 mask 全部错位（bool err=0.18, float 超容差）——
   生成代码"签名对、语义浅"的典型断层
4. **诊断插曲（方法论记录）**: 我们的 PATCH-2 曾把循环从偏移语义改成
   计数语义但没同步乘 BLOCK_N，制造了"第二 K 块丢失"的假象——
   假设检验法（对照"只算前 64 K"的构造输出，err=0.019 vs 其他假设
   0.16）三步定位。教训: **调试别人的代码时，先怀疑自己的 patch**

## 归因分解（谁贡献了什么）

| 差距来源 | 量化 |
|---|---|
| KernelGen 平台优化管线 vs 本地裸 LLM | 1.93ms（2.94→1.52）: 尾块/causal 截断等提示消费 |
| 手写 vs KernelGen 官方 | 0.09ms（1.52→1.44）+ mask 4D 语义 + 平台内置健壮性 |
| 人工 patch 投入 | 2 处（codegen 兼容 ×1 + 循环偏移自伤修复 ×1） |
| 原生 CANN 差距 | 双方同为 ~6.2-6.6x（栈水位，见 perf_analysis） |

## 结论

1. **官方 KernelGen 确实有货**: 生成质量明显高于裸 LLM（提示词被
   消费、结构完整），性能达手写版 94%（1.06x）
2. **但"开箱可用"仍不成立**: 跨芯片方言编译失败 ×1 + mask 语义
   缺陷 ×2——我们的四阶段验证每一关都拦到了真问题
3. **手写版的剩余价值**: 4D mask 完整语义、平台健壮性内置、
   以及**验证体系本身**（本次实验的全部发现都由它产出）
4. 与 §8.3 论文数据一致: attention 类生成的"高不成低不就"——
   60-80% 正确率带 1-2 个语义断层，正是需要验证层把守的原因

## 产物

- `mcp_result.json` 官方原始返回 · `kernelgen_triton_orig.py` 未改原版
- `kernelgen_triton.py` patch 版（PATCH-1/2d/4/5 标注）
- `compare_kernelgen_vs_hand.py` / `debug_kg*.py` 诊断链
- 复现: `python3 intake_llm/compare_kernelgen_vs_hand.py`
