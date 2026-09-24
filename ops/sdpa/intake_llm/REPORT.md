# LLM-Track 生成版 vs 手写版对比实验（本地复现 KernelGen 方法论）

日期: 2026-09-17 · 方法: 按 flagos-skills/kernelgen-flagos 的
generate_kernel 参数协议单轮生成（LLM Track，无迭代修复），
公平性约束: 生成时不含开发中获得的本机平台知识。

## 协议还原

- `func_type: attention`——KernelGen 官方对 attention 类标注**成功率 ~60%**
  并要求用户确认（skill 文件的 Complexity Gate 原文）
- `flagos_wiki` 按 skill 规则构造: kernel pattern 首条 + Triton anchor
  第二条 + 内存布局/访存提示
- 生成产物: `intake_llm/kernel_llm_track.py`（flash-attention v2 教科书
  布局——生成模型的默认先验）
- 验证: FlagOS-OP intake 四阶段（编译/正确性/哨兵/性能）

## 结果

| 阶段 | 结果 | 细节 |
|---|---|---|
| ① 编译运行 | ✅ | 一次通过（64×64 tile 恰好避开 UB 上限——运气而非知识） |
| ② 正确性 | **3/5** | basic-causal/batch/d128 过；**tail100 err=1.9**（无尾块 masked load，越界读）；**noncausal err=1.5**（causal mask 硬编码，is_causal=False 也生效） |
| ③ 哨兵 | ✅ | 确定性+输入敏感（静默 no-op 未触发——但功能性 bug 哨兵不管，正确性阶段已拦） |
| ④ 性能 | 2.935ms | **13.3x native / 2.06x 慢于手写**（全量遍历 causal——正是手写版 V4 截断修掉的那 1.93x） |

## 与 KernelGenBench 论文数据的互相印证

- 论文: attention 类在 Pass@K 上显著弱于 pointwise/blas 类，
  Agent Track 88-96% 的平台准确率主要靠**迭代修复**堆出来
- 本实验: 单轮生成 3/5（60%）——**与官方对 attention 类 ~60% 的
  标注精确一致**
- 两个失败模式恰是"教科书先验 vs 平台现实"的断层:
  1. 尾块越界（教科书写法默认 S 整除 BLOCK——FlagOS-OP #15a 同款）
  2. is_causal 参数被忽略（签名有了、语义没接——生成模型常见断层）
- 性能差 2.06x ≈ 手写版 V4 截断的 1.93x + 尾块检查开销——
  **手写版的增量恰好就是"跑过实验才知道的平台/写法知识"**

## 结论

1. **LLM 单轮生成的起点是"能编译、主路径对、性能 60-70% 水位"**——
   可用性真实存在，但直接交付不够
2. **手写版的价值恰好补在 LLM 盲区**: 尾块安全（36 case 全过）、
   完整签名语义（mask/GQA/scale）、写法粒度（causal 截断 1.93x）、
   平台坑（dot dtype/ieee/exp2——本实验生成版恰好没踩到但也没防护）
3. **验证层不可替代的实证**: 正确性 3/5 是"看起来能用"，intake 四阶段
   把它拦在交付门外——这正是 FlagOS-OP intake 通道设计的意义
4. 若走 Agent Track（失败信息回喂迭代），tail100/noncausal 两个 bug
   都是机器可读的失败原因，预期 2-3 轮可收敛——但每轮都消耗
   编译+验证周期，人工效率在"平台知识已沉淀"时仍然占优

## 复现

```bash
python3 intake_llm/compare_llm_vs_hand.py    # 全流程 < 5 分钟
```
