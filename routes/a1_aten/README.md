# 路线 A1: Triton 算子 → torch dispatcher (aten) 注册

## 适用场景

替换 torch 已有算子（add / softmax / gelu / silu ...）的实现。
注册后**任何框架代码**（vLLM / transformers / 自研脚本）调用
`torch.xxx` 都会自动走你的 Triton kernel，无需改业务代码。

## 原理

```python
lib = torch.library.Library("aten", "IMPL")     # 挂载 aten 实现库
lib.impl("gelu", my_triton_gelu, dispatch_key)  # key 来自设备 profile
```

这是 PyTorch 官方扩展机制（FlagGems.enable() 内部即此机制），
不是 monkey patch。

## 文件

| 文件 | 作用 |
|---|---|
| `kernels.py` | Triton kernel 模板（gelu tanh 版） |
| `register_aten.py` | aten 注册模板（gelu Triton / silu 数值恒等计数） |

## 测试入口

```bash
python3 run.py --route a1 --level op --device <profile名>   # 示例: p800-kunlunxin
python3 run.py --route a1 --level framework --device <profile名>
```

## 注意事项

1. dispatch key 来自设备 profile（如 CUDA / PrivateUse1 / CPU）。
2. 同一 (算子, dispatch key) 重复注册会覆盖——与 `flag_gems.enable()`
   同进程使用时注意白名单避免冲突。
3. CPU 张量不受影响（只注册了设备 key），可用于精度参考。
