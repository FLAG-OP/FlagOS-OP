# 算子集成指南: 开发完成之后怎么交付

[← 返回文档中心](index.md)

> 这一页回答两个开发后期的问题: **我的算子应该交到哪儿**（FlagGems、
> aten 注册、vllm_fl 插件还是厂商身份），以及**走框架分发要付多少
> 开销**——后者有本机实测数字，不靠推测。

## 两张调度网（先纠正一个常见误解）

FlagGems **本身不是调度器**——它是一个 kernel 集合，没有自己的分发
机制。真正的统一调度有两张网，而 FlagGems 同时出现在两张网里:

```
调用方（vLLM / 业务代码）
 │
 ├─ torch.gelu / torch.softmax …（标准 aten 算子）
 │    → 【网一】PyTorch aten dispatcher
 │         └─ flag_gems.enable() 把 ~200 个 Triton kernel 注册到这里
 │            （A1 机制；贡献进 FlagGems 的算子走这张网全局生效）
 │
 └─ call_op("silu_and_mul"/"rms_norm"/…)（vLLM 融合算子）
      → 【网二】vllm_fl OpManager（A2/B 的统一调度网）
           ├─ default.flagos  ← vllm_fl 的 flaggems backend，
           │                      把 FlagGems 融合算子实现包装注册
           │                      （实测源码: backends/flaggems/）
           ├─ vendor:xxx        ← B 路线厂商实现（ascend/cuda/iluvatar…）
           └─ reference.torch   ← 数值参考
```

所以准确的说法是: **A2/B 的统一调度网是 vllm_fl OpManager，而
FlagGems 的融合算子实现已经在这张网里**——身份是 `default.flagos`
（priority 150），你的 A2/B 实现注册后与它同网竞争，靠
`with_allowed_vendors` / `VLLM_FL_PER_OP` 钉选。不能做的是把 A2/B
算子"注册进 FlagGems"——FlagGems 是静态 kernel 集合，不消费外部
注册。另外，若某个厂商 kernel 实现的是标准 aten 算子，它可以走 A1
进【网一】——同一个 kernel 允许在两张网里各有一个身份。

## 四条交付路径怎么选

选择只取决于一个问题: **你的算子的宿主是谁**。

| 宿主 | 交付路径 | 机制 | 详解 |
|---|---|---|---|
| torch 通用算子（add/gelu/softmax…） | **贡献进 FlagGems**（或本地 aten 注册） | `torch.library.Library("aten","IMPL")` | [A1 路线](route-a1-aten.md) |
| vLLM 融合算子（silu_and_mul/rms_norm…） | **vllm_fl 插件** | `VLLM_FL_PLUGIN_MODULES` + OpImpl | [A2 路线](route-a2-dispatch.md) |
| 厂商专有 kernel | **厂商身份注册** | Backend + OpImpl(VENDOR) | [B 路线](route-b-vendor.md) |
| 自用/实验 | 样板目录直接跑 | 无注册，test/ 三层 | [算子样板](../templates/operator/README.md) |

### 要不要贡献进 FlagGems？

**如果你的算子是标准 torch 算子的更好实现，FlagGems 是正确的归宿。**
理由有三: 一是机制天然吻合——FlagGems 内部就是 A1 的 aten 批量注册
（`enable()` 一次注册约 200 个算子），你贡献的 kernel 进入后随
`flag_gems.enable()` 全局生效，任何框架代码调 `torch.xxx` 自动受益；
二是社区维护，多芯片 CI 替你兜底；三是本库的三层验证体系产出
（精度数据、性能基线、哨兵结论）正好是上游 PR 需要的证据。

反过来，**vLLM 融合算子不是"进不了统一调度"**——它们的宿主是
vllm_fl OpManager（网二），而且 FlagGems 的融合算子实现本来就在
这张网里当 `default.flagos`。你贡献一个融合算子到 FlagGems 后，
vllm_fl 的 flaggems backend 适配层（`backends/flaggems/impl/`）
可以为它做包装接入；但若适配层没覆盖你的算子，直接写 A2 插件注册
更省事。厂商专有 kernel（依赖特定 SDK）不适合进通用库，走 B——
但注意 B 与 FlagGems 同在网二，是竞争关系而非互斥。

## 分发开销（P800 实测）

```bash
python3 scripts/bench_dispatch.py --route a1   # aten 路径
python3 scripts/bench_dispatch.py --route a2   # call_op 路径
```

| 分发路径 | 实测开销 | 相对占比* | 结论 |
|---|---|---|---|
| **A1 aten 注册**（=FlagGems 机制） | **≈3µs** | 6% | C++ dispatcher，可忽略 |
| **A2 call_op** | **≈17-21µs** | 31-41% | Python 策略层，微 kernel 需注意 |
| 直调 kernel（baseline） | ~50µs | — | 本栈启动 floor 本身就不低 |

\* 占比以 ~50µs 的 pointwise kernel 为基数。**换算到真实场景**:
LLM 推理的主力 kernel（attention/GEMM）耗时 100µs~10ms，A2 的
20µs 在 100µs 算子上是 20%、在 1ms 算子上仅 2%——越大的算子，
分发开销越不值得纠结。

三条实用结论:

1. **A1/FlagGems 路径无开销顾虑**（3µs），放心走框架
2. **A2 的 20µs 只对微小 pointwise 算子有感**——若某个微算子处于
   热循环且本机 `enforce_eager`（无 CUDA graph 摊薄），可考虑该点
   直调或走 A1，其余照常 dispatch
3. 一个此前被忽视的事实: 本栈**启动 floor ~50µs** 比两条分发开销都大
   （64×1024 与 8192×8192 直调同为 ~52µs），优化微小算子时先看
   启动成本再谈分发

## 交付 checklist

开发完成（[全链路指南](fullstack-guide.md) 七步走完）后，交付前确认:

- [ ] 精度: `script/check_accuracy.py` 全过（含非整倍数 shape 与
      zeros/large/boundary 特殊用例）
- [ ] 哨兵: 确定性 + 输入敏感（防 #11/#15 类静默缺陷）
- [ ] 性能: 挂入 `perf_registry` 过回归门禁；与原生/FlagGems 三方对比
- [ ] 分发开销: 对微小算子跑一次 `bench_dispatch.py` 确认可接受
- [ ] 报告: `REPORT.md` 总报告 + 分册齐备（[样例](../reports/examples/)）
- [ ] 交付: 按上表选路径；贡献 FlagGems 时把精度/性能数据附进 PR

---

**下一步**: [已知问题](known-issues.md)——集成到真实栈之前的最后
一道排雷清单。
