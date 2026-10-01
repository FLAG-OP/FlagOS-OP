# 端到端基准: mini-LLM 双算子消费（embedding + sdpa）vs 全原生

> 模型: 4 层 Llama 风格 mini-LLM（GQA 8/2 · causal · vocab 32k ·
> d=512 · hidden 2048），业务代码只调 `nn.Embedding` /
> `F.scaled_dot_product_attention`（标准 aten 路径，A1/patch 后自动接管）。
> 脚本: `e2e_bench.py`（子进程隔离三链，防注册污染）。
> 日期: 2026-10-01 · fp16 · npu:0。

## 结果（端到端每 forward 墙钟）

| 场景 | A 全原生 | **B 生产链** | B/A | C 纯 Triton | C/A |
|---|---|---|---|---|---|
| prefill S=1k B=8 | 4.25ms | **4.19ms** | **0.99x** | 9.36ms | 2.20x |
| prefill S=2k B=4 | 4.71ms | **4.65ms** | **0.99x** | 15.05ms | 3.20x |
| prefill S=4k B=2 | 5.30ms | **5.26ms** | **0.99x** | 26.41ms | 4.99x |
| decode B=32 (cache 511) | 1.87ms | 2.10ms | 1.12x | 2.26ms | 1.21x |

B 生产链 = sdpa 智能路由（大 S 截流原生）+ embedding 厂商委托。

## 三个结论

1. **生产链端到端零损耗（0.99x）**——prefill 全档 B 链与全原生在
   测量噪声内不可区分。截流路由 + 委托架构在真实模型里兑现了设计
   目标：业务代码零改动拿到原生水位
2. **纯 Triton 链的端到端代价 2.2-5.0x**——单算子 12x 的差距被模型
   其余部分（GEMM/LN/MLP 占大头）稀释到端到端 2-5x；decode 场景
   两链接近（attention 占比小）。这是"截流必要性"的端到端证据
3. **decode B 链 1.12x**——小 S 路由进 Triton（阈值 1024 之下）的
   代价在 decode 场景可见（1.87→2.10ms）。若 decode 为热路径，
   可将 `SDPA_DISPATCH_S` 降到 256 或直接 `SDPA_DISPATCH_MODE=native`

## 过程中修复的真实 bug（同进程多算子场景）

e2e 首次把两个算子装进同一进程，暴露了三个交付缺陷（均已修复并
本地回归全绿，待同步 PR）：
1. `auto_dispatch` 内 `import register` 裸导入被 embedding 同名模块
   遮蔽（AttributeError）→ 文件路径锚定 + 单例缓存
2. sdpa `register.py` 内 `from kernel.* import` 同病（embedding 的
   kernel/ 目录遮蔽）→ `_load_kernel_mod_mod` 锚定加载器
3. 属性版/模块版加载器各 exec 新实例导致 stats/patch 状态分裂
   （stats 全 0）→ 统一单例（sys.modules 注册）

教训沉淀: **FlagOS-OP 每算子独立目录 + 顶层名（register/kernel）
相同的结构，在"多算子同进程消费"场景必然冲突**——锚定加载是当下
的修复，长期方案是算子目录包化（ops/sdpa/ 作为包导入）或库名
唯一化，建议在库层面立项（多算子 e2e 消费是框架层验证的前置）。

## 复现

```bash
python3 e2e_bench.py     # 三链全跑（~10 分钟，含 Triton 编译）
```
