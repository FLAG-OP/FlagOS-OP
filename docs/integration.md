# 算子集成指南: 开发完成之后怎么交付

[← 返回文档中心](index.md)

> 这一页回答两个开发后期的问题: **我的算子应该交到哪儿**（FlagGems、
> aten 注册、vllm_fl 插件还是厂商身份），以及**走框架分发要付多少
> 开销**——后者有本机实测数字，不靠推测。

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

反过来，**vLLM 融合算子不适合进 FlagGems**——它们不是 aten 算子，
torch dispatcher 管不到，宿主在 vllm_fl 的 dispatch 体系里，走 A2。
厂商专有 kernel（依赖特定 SDK）也不适合进通用库，走 B。

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
