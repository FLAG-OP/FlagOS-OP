"""``aten::sigmoid`` 的高精度语义参考实现。"""
from __future__ import annotations

import torch

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def sigmoid_reference(x: torch.Tensor) -> torch.Tensor:
    """返回 ``1 / (1 + exp(-x))``，浮点低精度输入在 fp32 中计算。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"sigmoid expects a Tensor, got {type(x)!r}")
    if x.is_complex() or x.dtype == torch.float64:
        return 1.0 / (1.0 + torch.exp(-x))
    out = 1.0 / (1.0 + torch.exp(-x.float()))
    return out.to(x.dtype) if x.is_floating_point() else out
