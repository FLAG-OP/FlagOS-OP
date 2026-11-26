# torch 级实现: ATen 组合表达 layer_norm（与 reference 同式）。
#
# 递归规避: 组合走 mean/pow/rsqrt/mul/add 基础算子，不触碰
# layer_norm/native_layer_norm 注册面。
from __future__ import annotations

import torch


def layer_norm_torch(input: torch.Tensor, normalized_shape, weight, bias,
                     eps: float = 1e-5) -> torch.Tensor:
    """与 reference 同语义的 ATen 组合。"""
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
    return y.to(input.dtype)
