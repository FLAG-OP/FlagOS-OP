# 路线 A2: FlagOS 融合算子

> 典型 kernel 层级: TR / FW——路线只管接入，不限定层级。

[← 返回文档中心](index.md): Triton → FlagOS 融合算子

## 适用场景

vLLM **融合算子**（silu_and_mul / rms_norm / rotary_embedding /
自定义融合层）。它们不是 aten 算子，torch dispatcher 管不到，
走 FlagOS 自研 dispatch（OpManager / OpRegistry / policy）。

## 分发模型

- 模型调用点（被 vllm_fl patch）→ `call_op("gelu_and_mul", ...)`
- OpManager（缓存/策略/fallback）→ OpRegistry
- OpRegistry: flagos(150) / vendor(100) / reference(50)

## 插件注册（不改 vllm-plugin-FL 源码）

```bash
export VLLM_FL_PLUGIN_MODULES=routes.a2_dispatch.plugin.register_ops
```

插件模块提供 `register(registry)` 函数，内部 `registry.register_many()`
注册 OpImpl 列表（每个算子至少 flagos + reference 双实现）。

## 开发步骤

1. 先写 PyTorch 参考实现**确定语义**
2. 写 Triton kernel 对齐参考（fp32 内部计算）
3. 双后端注册: `default.flagos`(150) + `reference.torch`(50)
4. 三段式测试: 精度（多 shape×dtype）/ `with_preference` 策略切换 / 性能
5. 框架级: 以 `vendor:<name>` 身份注册 + `VLLM_FL_PER_OP` 精确钉住

## 参考实现与样例

- 正式实现: `routes/a2_dispatch/`
- 自包含样例: `examples/a2-op/`（gelu_and_mul 全流程）
- 框架级样例: `examples/a2-framework/`（Triton silu_and_mul 注入真实推理）

## 注意事项

1. 插件入口函数必须是 `register` / `vllm_fl_register`
   （官方 README 写的 `register_builtins` 仅适用内置 backend）
2. 同进程串跑多个测试需 `reset_default_manager()`（OpManager 有缓存）
3. 框架级注入用 vendor 身份（`vendor:triton-template`）——
   PER_OP 的 token 只支持 flagos/vendor/reference/vendor:<name>，
   不能直接钉 impl_id
4. 自定义数值实现的框架级断言用"前 2 token 一致率"而非全量一致
   （随机权重 + 贪心解码会把数值微差混沌放大为后期分叉）
