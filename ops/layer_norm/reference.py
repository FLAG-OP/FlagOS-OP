# layer_norm 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::layer_norm(input, normalized_shape, weight=None, bias=None,
#                         eps=1e-5, cudnn_enable=True) -> Tensor):
#   - 公开 API，内部消费 native_layer_norm 原语（issue #55 要求两者
#     同时验证——本算子与 ops/native_layer_norm 配对交付，注册/测试
#     一一对应，见 ops/native_layer_norm/test/op_level.py 第 1b 步）。
#   - out: (input - mean) / sqrt(var + eps) * weight + bias。
#   - 与 native_layer_norm 的差异仅在返回值（公开 API 丢弃 mean/rstd）。
#
# 参考实现约定: 内部 fp32 计算、cast 回输入 dtype（acceptance.md）。
# 不依赖任何注册/分发（纯函数）。
from __future__ import annotations

import torch


def layer_norm_reference(input: torch.Tensor, normalized_shape, weight, bias,
                         eps: float = 1e-5) -> torch.Tensor:
    """aten::layer_norm 的 fp32 内部精度参考实现。"""
    if not isinstance(input, torch.Tensor):
        raise TypeError("layer_norm expects a Tensor")
    out, _, _ = _reference_tuple(input, normalized_shape, weight, bias, eps)
    return out


def _reference_tuple(input, normalized_shape, weight, bias, eps):
    dims = [-i for i in range(1, len(normalized_shape) + 1)]
    out_dtype = input.dtype
    xf = input.float()
    mean = xf.mean(dim=dims, keepdim=True)
    var = xf.var(dim=dims, keepdim=True, correction=0)  # 有偏方差（ATen 口径）
    rstd = 1.0 / torch.sqrt(var + eps)
    y = (xf - mean) * rstd
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    red_shape = tuple(xf.shape[: len(xf.shape) - len(normalized_shape)])
    return (y.to(out_dtype),
            mean.reshape(red_shape).to(out_dtype),
            rstd.reshape(red_shape).to(out_dtype))
