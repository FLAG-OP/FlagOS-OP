# 样例索引（3 路线 × 3 层级 = 9 格）

每个样例目录 = 一个矩阵格的完整可运行演示。

<a id="map"></a>
## 样例定位图（每个样例在矩阵中的位置与涵盖范围）

```mermaid
flowchart TB
    classDef tr_level fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef torch_level fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef hw_level fill:#ffedd5,stroke:#ea580c,color:#7c2d12
    classDef fs fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12

    subgraph L0["算子库层 · kernel 直测"]
        direction LR
        A1K["a1-kernel"]:::tr_level
        A2K["a2-kernel"]:::tr_level
        BK["b-kernel<br/>硬件+torch"]:::hw_level
    end
    subgraph L2["框架层 · op 注册/分发"]
        direction LR
        A1O["a1-op"]:::tr_level
        A2O["a2-op"]:::tr_level
        BO["b-op"]:::torch_level
    end
    subgraph L4["应用层 · framework"]
        direction LR
        A1F["a1-framework"]:::torch_level
        A2F["a2-framework"]:::tr_level
        BF["b-framework"]:::torch_level
    end

    HWK["hw-kernel-example<br/>硬件级·SDK模板"]:::hw_level
    BMM["bmm-fullstack ⭐<br/>A1 · Triton 级 · GEMM"]:::fs
    BFS["b-fullstack ⭐<br/>B · torch 级 · fused"]:::fs
    SMX["softmax-fullstack ⭐<br/>A1 · Triton 级 · reduction"]:::fs
    BWD["backward-example<br/>A2 · Triton 级 · autograd"]:::fs

    BMM -. "算子库层→框架层→应用层" .-> A1K
    BFS -. "算子库层→框架层→应用层" .-> BK
    SMX -. "算子库层→框架层→应用层" .-> A1K
    BWD -. "fwd+bwd" .-> A2K
```

**读图方式**:
- 每行 = 物理栈一层（算子库/框架/应用），行内左→右 = A1 / A2 / B 三条路线
- 想按**开发步骤**（而非矩阵格）找范本，见
  [全链路指南 · 步骤 × 样例覆盖矩阵](../docs/fullstack-guide.md#coverage)
- 节点配色 = [开发层级](../docs/architecture.md#levels):
  🔵 **Triton**（Triton 层，自研设备码）· 🟢 **torch**（框架层）· 🟠 **硬件**（硬件语言层）
- ⭐ 全链路/专项样例以虚线标注覆盖范围（bmm/b-fullstack/softmax 各覆盖
  对应列三层；backward-example 覆盖 A2 列的 fwd+bwd）
- 全链路/专项: bmm-fullstack(GEMM) / b-fullstack(fused) /
  softmax-fullstack(reduction+autotune) / backward-example(autograd)
- b-kernel 同时含 硬件（厂商 kernel 直测）与 torch（csrc JIT 编译闭环）


| 样例 | 内容 | 开发层级 |
|---|---|---|
| [a1-kernel](a1-kernel/) | Triton gelu 直测（精度+哨兵+性能） | **Triton** |
| [a1-op](a1-op/) | 从零写 Triton relu → aten 注册 → 拦截/精度/性能 | **Triton** |
| [a1-framework](a1-framework/) | aten::silu 恒等计数注入真实 vLLM 前向 | **torch**（恒等计数·torch 组合） |
| [a2-kernel](a2-kernel/) | Triton gelu_and_mul 直测 | **Triton** |
| [a2-op](a2-op/) | Triton gelu_and_mul → dispatch 双后端 → 三段式 | **Triton** |
| [a2-framework](a2-framework/) | Triton silu_and_mul 以 vendor 身份注入真实推理 | **Triton** |
| [b-kernel](b-kernel/) | **厂商 kernel 直测 + C++ JIT 编译闭环 + 哨兵检查** | **硬件 + torch** |
| [b-op](b-op/) | 自定义 厂商算子注册 注册/选择/计数 | torch（委托） |
| [softmax-fullstack](softmax-fullstack/) | 行归约 + autotune 三层 | **Triton** |
| [backward-example](backward-example/) | autograd fwd+bwd + 训练冒烟 | **Triton** |
| [hw-kernel-example](hw-kernel-example/) | xtorch_ops 厂商原语组合 + CUDA C++ 参考（P800 硬件级） | **硬件级** |
| [bmm-fullstack](bmm-fullstack/) | torch.bmm 贯穿 算子库层→框架层→应用层（含 [#11](../docs/known-issues.md) 根因发现） | **Triton** |
| **[b-fullstack](b-fullstack/)** ⭐ | **旗舰: 同一 C++ kernel 贯穿 算子库层→框架层→应用层**（JIT 编译→vendor 注册→真实推理，附[开发报告](b-fullstack/report.md)） | **torch** |
| [b-framework](b-framework/) | audit vendor 拦截真实 vLLM + 黄金回归 | torch（委托） |

运行方式统一: `python3 examples/<样例名>/example.py [设备profile名]`。

样例与正式测试（`run.py`）复用同一套基础设施（harness / golden / 设备 profile）；
kernel/op 层样例完全自包含（不依赖 routes/），可直接复制为开发起点。

<a id="perf"></a>
## 性能速览（参考实例实测）

| 算子 @ shape (bf16) | 最优实现 | 关键数字 | 详见 |
|---|---|---|---|
| silu_and_mul 4096×8192 | Triton 融合 | 0.083ms（C++ 0.271 / 参考 0.651） | [b-fullstack](b-fullstack/) |
| BMM 16×512³ | Triton 分块 | 0.030ms / 142 TFLOPS（原生 1.59x） | [bmm-fullstack](bmm-fullstack/) |
| gelu 8192² | Triton | 0.047ms（CPU 2341x） | [a1-op](a1-op/) |
| gelu_and_mul 8192² | Triton 融合 | 0.052ms（vs 分解参考 6204x） | [a2-op](a2-op/) |
| softmax 1024² | Triton 流式归约 | 见样例输出 | [softmax-fullstack](softmax-fullstack/) |
| gelu_and_mul bwd | Triton autograd | dx_err=1.7e-06 | [backward-example](backward-example/) |

> 所有数字: 健康态进程、短采样(≤100 次)。共享设备的进程内污染可致
> 200x 级失真（[known-issues](../docs/known-issues.md) #10/#11），勿跨进程直接对比。

一次性数字只作人读展示；可重复回归门禁见
[性能回归追踪](../docs/performance-regression.md)。

## AI 生成算子 intake

[KernelGen](../docs/ai-intake.md) 等来源产出的 kernel 经
[intake 契约](../intake/README.md) 自动进入三级验证:

| case | 说明 | P800 实测 |
|---|---|---|
| [kernelgen-gelu-example](../intake/cases/kernelgen-gelu-example/) | 正例: 生成 kernel + A2 注册 | PROMOTED |
| [kernelgen-gelu-no-device-context](../intake/cases/kernelgen-gelu-no-device-context/) | 负例: 静默 no-op | BLOCKED-OK（哨兵检出 [#11](../docs/known-issues.md)） |
