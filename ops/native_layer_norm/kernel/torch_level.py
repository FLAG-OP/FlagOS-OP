# torch 级实现: ATen 组合表达 native_layer_norm（mean/var/rstd 显式组合）。
#
# 递归规避: 组合走 mean/pow/sqrt/add/rsqrt/mul 等基础 ATen 算子，
# 不触碰 layer_norm/native_layer_norm 注册面。
from __future__ import annotations

import torch


def native_layer_norm_torch(input: torch.Tensor, normalized_shape, weight,
                            bias, eps: float = 1e-5):
    """与 reference 同语义的 ATen 组合，返回 (out, mean, rstd)。"""
    dims = [-i for i in range(1, len(normalized_shape) + 1)]
    xf = input.float()
    mean = xf.mean(dim=dims, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    y = (xf - mean) * rstd
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    red_shape = tuple(xf.shape[: len(xf.shape) - len(normalized_shape)])
    return (y.to(input.dtype),
            mean.reshape(red_shape).to(input.dtype),
            rstd.reshape(red_shape).to(input.dtype))
