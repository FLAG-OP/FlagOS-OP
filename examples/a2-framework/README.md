# 样例: A2 路线 × 框架层

## 目标

把 Triton 实现的自定义融合算子注入**真实 vLLM 推理**，
验证被真实调用且端到端基本不破坏。

## 注入机制

- Triton silu_and_mul 以 `vendor:triton-template` 身份注册
  （vendor 名可被 PER_OP 精确钉住，避免与内置 default.flagos 冲突）
- `VLLM_FL_PLUGIN_MODULES` 注入（每个 vLLM 子进程自动发现）
- `VLLM_FL_PER_OP="silu_and_mul=vendor:triton-template|reference"`

## 断言策略（自定义数值实现 ≠ 恒等）

1. 调用计数 > 0
2. 前 2 token 一致率 ≥ 2/3×N——随机权重 + 贪心解码下，
   任何数值微差都会被混沌放大为后期分叉（实测结论）

## 运行（约 2 分钟）

```bash
python3 examples/a2-framework/example.py
```

## 预期输出

```
triton silu calls  : >0
output compare     : N/6 prompts matched first 2 tokens
```

## 同算子贯穿三层

本路线三层样例各自独立可跑；若要跟踪**同一算子**走完
kernel→op→framework，把各层样例的算子替换为同一个即可。
gelu_and_mul 的贯穿实证见 `tests/kernel_level/test_consistency.py`
与旗舰样例 [examples/b-fullstack](../b-fullstack/)。
