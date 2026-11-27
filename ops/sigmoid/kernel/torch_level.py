"""由 PyTorch/ATen 基础算子组合得到的 sigmoid 基线实现。"""
from __future__ import annotations

import torch

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"sigmoid expects a Tensor, got {type(x)!r}")
    if x.is_complex() or x.dtype == torch.float64:
        return 1.0 / (1.0 + torch.exp(-x))
    out = torch.reciprocal(1.0 + torch.exp(-x.float()))
    return out.to(x.dtype) if x.is_floating_point() else out
