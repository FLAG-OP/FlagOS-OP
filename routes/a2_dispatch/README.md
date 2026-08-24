# 路线 A2: Triton 算子 → FlagOS dispatch 插件

## 适用场景

vLLM 融合算子（silu_and_mul / rms_norm / rotary / 自定义融合层）。
这些算子不是 aten 算子，torch dispatcher 管不到，需走 FlagOS 自研
dispatch（OpManager / OpRegistry / policy）。

## 接入方式（不改 vllm-plugin-FL 源码）

```bash
export VLLM_FL_PLUGIN_MODULES=routes.a2_dispatch.plugin.register_ops
```

## 注册结构

- `gelu_and_mul`: `default.flagos`(150, Triton) + `reference.torch`(50) 双实现
- `silu_and_mul`: `vendor.triton-template`（Triton 实现以 vendor 身份注册，
  便于 `VLLM_FL_PER_OP="silu_and_mul=vendor:triton-template|reference"` 精确钉住，
  避免与内置 default.flagos 冲突）

## 测试入口

```bash
python3 run.py --route a2 --level op --device <profile名>   # 示例: p800-kunlunxin
python3 run.py --route a2 --level framework --device <profile名>
```
