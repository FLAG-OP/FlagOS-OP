# 全链路开发指南: 一个算子从源码到真实推理

[← 返回文档中心](index.md)

> 这一页把开发一个算子拆成七步：定义语义、实现 kernel、直测、
> 注册、注入真实推理、跨层一致性、性能与报告。每一步先讲要做什么、
> 做到什么程度算完成，再按开发级别和路线指向对应的可运行样例。
> 走完全部步骤，你会得到一个通过三层验证、进入性能基线、
> 带开发报告的算子。

<a id="pipeline"></a>
## 流水线总览

```mermaid
flowchart LR
    S0["第0步<br/>定义语义"] --> S1["第1步<br/>实现 kernel"]
    S1 --> S2["第2步<br/>kernel 层直测"]
    S2 --> S3["第3步<br/>注册进框架"]
    S3 --> S4["第4步<br/>应用层注入"]
    S4 --> S5["第5步<br/>跨层一致性"]
    S5 --> S6["第6步<br/>性能基线+报告"]
```

| 步 | 做什么 | 完成标志（DoD） |
|---|---|---|
| 0 | 定义算子语义与数值规格 | 有参考实现 + 容差 + 可复现输入 |
| 1 | 按开发级别实现 kernel | kernel 可编译/可调用（无注册） |
| 2 | kernel 层直测 | 精度 ✓ · 哨兵 ✓ · 性能有数字 |
| 3 | 按路线注册进框架 | 钉选调用命中你的 impl_id |
| 4 | 注入真实推理 | 调用计数 > 0 · 输出比对通过 |
| 5 | 跨层一致性 | 直调输出 ↔ dispatch 输出张量级一致 |
| 6 | 收尾 | perf 基线更新 · 报告生成 |

三个维度在流水线上的位置: **路线**（A1/A2/B）决定第 3~4 步的接入机制，
**开发级别**（torch/Triton/硬件级）决定第 1 步的写法，
**验证层级**（算子库层/框架层/应用层）就是第 2/3/4 步本身。

---

## 第 0 步: 定义算子语义（先定判卷标准，再写实现）

**抽象动作**（与路线/芯片无关）:

1. 写 **PyTorch 参考实现**——它是后续所有层级的判卷标准
2. 定 **接口签名**（输入张量数/shape/dtype/返回或 out 参数）
3. 定 **数值容差**（fp32 通常 1e-5，bf16/fp16 通常 1e-2；归约/GEMM
   用相对容差）
4. 定 **可复现输入**（seed + shape + dtype；框架级另备 prompt 集）

**范本**:

| 场景 | 看哪里 |
|---|---|
| 融合算子参考实现 | [a2-op](../examples/a2-op/)（`impl_reference` 与 Triton 同文件对照） |
| GEMM 用宿主算子当参考 | [bmm-fullstack](../examples/bmm-fullstack/)（`torch.bmm` 即语义锚） |
| 归约算子 | [softmax-fullstack](../examples/softmax-fullstack/)（fp32 归一化参考） |
| 正/反向成对定义 | [backward-example](../examples/backward-example/)（fwd + bwd 双参考） |
| 语义锚点库 | `common/kernel_spec.py` 的 `SEMANTIC_REFS` |
| 声明式输入模板 | [inputs/spec.yaml](../inputs/spec.yaml) + [gen_inputs](testing.md#inputs) |

---

## 第 1 步: 实现 kernel（三级别选型）

**抽象动作**: 选择开发级别 → 写 kernel → 需要时 JIT 编译。
级别与路线正交（[体系结构](architecture.md#levels)）。

| 级别 | 写什么 | 编译/执行 | 范本 |
|---|---|---|---|
| **torch 级** | C++ extension 调 ATen / Python 委托 | `cpp_extension.load()`，无新设备码 | [b-fullstack](../examples/b-fullstack/)（C++ 调 ATen）· [b-op](../examples/b-op/) |
| **Triton 级** | `@triton.jit` / `@pointwise_dynamic` | Triton → 芯片编译栈 → 设备码 | [bmm-fullstack](../examples/bmm-fullstack/)（GEMM）· [softmax-fullstack](../examples/softmax-fullstack/)（归约+autotune）· [a1-kernel](../examples/a1-kernel/) · [a2-kernel](../examples/a2-kernel/) |
| **硬件级** | 厂商语言设备码 / 厂商预编译原语 | 厂商工具链 → `.so` | [hw-kernel-example](../examples/hw-kernel-example/)（xtorch_ops 原语 + [SDK 模板](../examples/hw-kernel-example/sdk_template/BUILD.md)）· [b-kernel](../examples/b-kernel/)（厂商 kernel 直测清单） |

**已有 AI 生成产物?** 直接走 [intake 契约](ai-intake.md)，
第 1 步由 KernelGen/KernelBench 产出，你从第 2 步的自动验证接手。

> ⚠️ Triton 级必读: 裸 `@triton.jit` 启动必须包
> `torch_device_fn.device(...)` 上下文，否则首次后的启动**静默 no-op**
> （[known-issues #11](known-issues.md)）。本库全部样例已包。

---

## 第 2 步: kernel 层直测（不经任何注册）

**抽象动作**——三个子项全部通过才算完成:

1. **精度**: vs 第 0 步的参考实现，多 shape × dtype
2. **[哨兵](testing.md#sentinel)**: 同输入两次调用一致（确定性）+
   输入变化输出变化（敏感）+ out 模式预填哨兵（真写入）
3. **性能**: 短采样（≤100 次）；需要回归门禁时挂入
   [perf_run](performance-regression.md)（每用例子进程隔离）

| 子项 | 范本 |
|---|---|
| 精度三件套 | [a1-kernel](../examples/a1-kernel/) · [a2-kernel](../examples/a2-kernel/) |
| out 参数哨兵（预填检测） | [b-kernel](../examples/b-kernel/)（检出 3 个厂商坏 kernel） |
| return 模式哨兵 | [a2-kernel](../examples/a2-kernel/) |
| 反向 kernel 验证 | [backward-example](../examples/backward-example/)（vs PyTorch autograd） |
| 性能挂基线 | `tests/kernel_level/test_*.py` 的 `perf_cases()`（[新增方法](performance-regression.md)） |
| AI 生成 kernel 自动直测 | [intake 正/负例](ai-intake.md) |

入口: `python3 run.py --route <a1|a2|b> --level kernel --device <profile>`

---

## 第 3 步: 注册进框架（三路线选型）

**抽象动作**: 把 kernel 挂到**目标算子宿主**的分发机制上，
然后验证"钉选命中 + fallback 正确"。

| 路线 | 宿主 | 机制 | 范本 |
|---|---|---|---|
| [A1](route-a1-aten.md) | torch aten 算子 | `torch.library.Library("aten","IMPL").impl()` | [a1-op](../examples/a1-op/)（relu 从零接管）· [bmm-fullstack](../examples/bmm-fullstack/)（bmm 拦截）· [softmax-fullstack](../examples/softmax-fullstack/) |
| [A2](route-a2-dispatch.md) | vLLM 融合算子 | dispatch 插件双后端（default+reference） | [a2-op](../examples/a2-op/) · [a2-framework](../examples/a2-framework/) |
| [B](route-b-vendor.md) | 以厂商身份提供 | `Backend` 子类 + `OpImpl(VENDOR)` | [b-op](../examples/b-op/) · [b-fullstack](../examples/b-fullstack/)（vendor:my-cpp） |

**验证要点**:
- 钉选: `with_preference(...)` + `with_allowed_vendors(...)`，或
  `VLLM_FL_PER_OP="op=vendor:xxx|reference"`
- 插件入口函数名是 `register` / `vllm_fl_register`
  （[文档与代码差异](known-issues.md)）

入口: `python3 run.py --route <X> --level op --device <profile>`

---

## 第 4 步: 注入真实推理（应用层）

**抽象动作**: 让 vLLM **子进程**发现你的插件 → 基线/插件双跑同
prompt 同 seed → 断言"被调用 + 不破坏输出"。

| 注入机制 | 适用 | 说明 |
|---|---|---|
| `VLLM_FL_PLUGIN_MODULES` | A2 / B | 每个 vLLM 子进程自动加载插件 |
| `VLLM_FL_PER_OP` | A2 / B | 只钉目标算子，其余走默认 |
| [sitecustomize 桥](route-a1-aten.md) | A1 | v1 前向在子进程，主进程 aten 注册不传播 |

**断言策略**（[详解](testing.md#assertions)）:
数值恒等实现 → 逐 prompt 前缀完全一致；自定义数值实现 →
前 2 token 一致率 ≥ 2/3×N，前缀按[黄金](testing.md#golden)自适应。

| 范本 | 演示的注入变体 |
|---|---|
| [a1-framework](../examples/a1-framework/) | aten 恒等计数 + sitecustomize |
| [a2-framework](../examples/a2-framework/) | vendor 身份 + PER_OP |
| [b-framework](../examples/b-framework/) | audit vendor + 黄金回归 |
| [b-fullstack](../examples/b-fullstack/) | C++ vendor 全链路第 3 幕 |
| [bmm-fullstack](../examples/bmm-fullstack/) | mini-attention 显式消费 bmm |
| [softmax-fullstack](../examples/softmax-fullstack/) | attention scores 消费 softmax |

入口: `python3 run.py --route <X> --level framework --device <profile>`

---

## 第 5 步: 跨层一致性

同算子同输入，在 **算子库层直调 ↔ 框架层 dispatch** 张量级比对，
证明"注册没有改变语义":

```bash
python3 run.py --consistency --device <profile>
```

锚定算子 gelu_and_mul，4×4 全零误差矩阵（参考实例）。
人工跟读版见各 fullstack 样例的逐幕输出；自动化实现见
[tests/kernel_level/test_consistency.py](../tests/kernel_level/test_consistency.py)。

---

## 第 6 步: 性能基线与报告

```bash
python3 scripts/perf_run.py  --device <profile> --update-baseline  # 有意更新时
python3 scripts/perf_compare.py --device <profile>                 # 回归门禁
python3 scripts/gen_report_scaffold.py --op <算子> --route <路线> --device <profile>
```

- 性能: 结构化记录 + 入库基线 + 20%/30% 双档门禁
  （[性能回归追踪](performance-regression.md)）
- 报告: 7 章骨架自动填环境与矩阵状态（[报告指南](reporting.md)）
- 完整报告范本: [b-fullstack 开发报告](../examples/b-fullstack/report.md)

---

<a id="coverage"></a>
## 步骤 × 样例覆盖矩阵

● 完整覆盖 · ◐ 部分/相关 · — 不涉及。找范本时先在此表定位，
再进对应样例读实现。

| 样例 | 0 语义 | 1 实现 | 2 kernel 层 | 3 注册 | 4 应用层 | 5 一致性 | 6 基线/报告 |
|---|---|---|---|---|---|---|---|
| [a1-kernel](../examples/a1-kernel/) | ● | ● Triton | ● | — | — | — | — |
| [a1-op](../examples/a1-op/) | ● | ● Triton | ● | ● aten | — | — | — |
| [a1-framework](../examples/a1-framework/) | ◐ | ◐ torch | — | ● aten | ● | — | — |
| [a2-kernel](../examples/a2-kernel/) | ● | ● Triton | ● | — | — | ◐ | — |
| [a2-op](../examples/a2-op/) | ● | ● Triton | ● | ● dispatch | — | — | ◐ 性能对比 |
| [a2-framework](../examples/a2-framework/) | ◐ | ◐ Triton | — | ● vendor | ● | — | — |
| [b-kernel](../examples/b-kernel/) | ● 语义锚 | ● 硬件+torch | ● 哨兵强化 | — | — | — | ◐ 性能 |
| [b-op](../examples/b-op/) | ◐ | ● torch | ◐ | ● vendor | — | — | — |
| [b-framework](../examples/b-framework/) | ◐ | ◐ | — | ● audit | ● 黄金 | — | — |
| **[b-fullstack](../examples/b-fullstack/)** ⭐ | ● | ● torch | ● | ● | ● | ◐ | ● 报告 |
| **[bmm-fullstack](../examples/bmm-fullstack/)** ⭐ | ● | ● Triton | ● | ● aten | ● | ◐ | ◐ 性能 |
| [softmax-fullstack](../examples/softmax-fullstack/) | ● | ● Triton | ● | ● aten | ● | ◐ | ◐ 性能 |
| [backward-example](../examples/backward-example/) | ● 正反向 | ● Triton | ● 含 bwd | ◐ autograd | — | — | ◐ 性能 |
| [hw-kernel-example](../examples/hw-kernel-example/) | ◐ | ● 硬件级 | ● | — | — | — | ◐ 性能 |
| [intake 正例](../intake/cases/kernelgen-gelu-example/) | ● manifest | ● 生成产物 | ● | ● A2 | — | — | ◐ perf 记录 |

## 按需求快速选范本

| 我想… | 直接看 |
|---|---|
| 最快跑通一条全链路（约 10 秒） | [bmm-fullstack](../examples/bmm-fullstack/) |
| 看 C++/vendor 全链路 + 完整开发报告 | [b-fullstack](../examples/b-fullstack/) |
| 学 aten 拦截（业务代码零改动） | [a1-op](../examples/a1-op/) → [a1-framework](../examples/a1-framework/) |
| 学 FlagOS dispatch 插件 | [a2-op](../examples/a2-op/) → [a2-framework](../examples/a2-framework/) |
| 学厂商 kernel 直测与哨兵 | [b-kernel](../examples/b-kernel/) · [hw-kernel-example](../examples/hw-kernel-example/) |
| 学训练侧（autograd fwd+bwd） | [backward-example](../examples/backward-example/) |
| 接入 KernelGen 生成产物 | [AI 生成算子接入](ai-intake.md) |

## 常见坑（全链路视角）

1. vLLM v1 前向在**子进程**——主进程注册不传播（A1 需
   [sitecustomize 注入](route-a1-aten.md)）
2. 自研 kernel 与[黄金](testing.md#golden)路径不同时，输出会混沌分叉
   （随机权重放大数值微差）——用一致率断言，不追位级相等
3. 性能基准长循环触发分配器池增长；短采样单独也不够，多用例需
   子进程隔离（[known-issues #10](known-issues.md)）
4. 计数文件要 pid 分片（TP 多 worker 并发写竞争，否则计数低估 ~35x）
5. 裸 Triton 启动缺 device 上下文 → 静默 no-op
   （[known-issues #11](known-issues.md)，intake 哨兵可自动拦截）

---

**下一步**: [测试体系](testing.md)——理解每层验证的断言策略与工具。
