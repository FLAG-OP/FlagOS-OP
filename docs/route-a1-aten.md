# 路线 A1: torch 算子替换

> 典型 kernel 层级: Triton（Triton）/ torch（torch 算子组合）——路线只管接入，不限定层级。

[← 返回文档中心](index.md): Triton → torch torch 算子替换

## 适用场景

替换 torch **已有算子**（add / softmax / gelu / silu / relu ...）的实现。
注册后任何框架代码（vLLM / transformers / 自研脚本）调用 `torch.xxx`
都自动走你的 kernel，无需改业务代码。

## 注册机制

```python
lib = torch.library.Library("aten", "IMPL")     # 挂载 aten 实现库
lib.impl("relu", my_triton_relu, dispatch_key)  # key 来自设备 profile
```

这是 PyTorch 官方扩展机制（`FlagGems.enable()` 内部即此机制，
批量注册 ~200 个算子），**不是 monkey patch**。

## 开发步骤

1. 写 Triton kernel（`@pointwise_dynamic` + `@triton.jit`）
2. 按 aten 签名包装（如 `relu(Tensor self) -> Tensor`）
3. 注册到设备 profile 指定的 dispatch key
4. 算子层三段式验证（拦截 / 精度 vs CPU 参考 / 性能）
5. 框架级注入验证（见下）

## 框架级注入三要素（关键难点）

vLLM v1 的前向跑在 **EngineCore 子进程**，主进程的 torch.library
注册不会传播，需要:

| 要素 | 做法 |
|---|---|
| 跨进程注册 | `injection/sitecustomize.py` 加入 PYTHONPATH，每个子进程启动即注册 |
| 防覆盖 | `VLLM_FL_FLAGOS_BLACKLIST=silu,silu_`（flag_gems.enable 也会注册同名算子） |
| 迫使命中 | `VLLM_FL_PER_OP=silu_and_mul=reference` 让模型走 F.silu → aten::silu |

## 参考实现与样例

- 正式实现: `routes/a1_aten/`
- 自包含样例: `examples/a1-op/`（从零写 relu）
- 框架级样例: `examples/a1-framework/`

## 注意事项

1. `Library` 对象必须保持引用，否则注册被回收
2. 同一 (算子, dispatch key) 重复注册会**覆盖**
3. CPU 张量不受设备 key 影响 → 天然的精度参考
4. 性能基准用短采样（100 次内），长循环会触发输出分配器池增长失真

本路线典型开发级别: **Triton 级**（自研设备码）或 **torch 级**（ATen 组合）

> ⚠️ 写 Triton kernel 时: 启动必须包 `torch_device_fn.device(...)` 上下文，
> 否则首次后的启动静默 no-op（[known-issues #11](known-issues.md)）。

---

**下一步**: [A2 路线](route-a2-dispatch.md)——FlagOS 融合算子的开发方式。 · [全链路指南](fullstack-guide.md)
