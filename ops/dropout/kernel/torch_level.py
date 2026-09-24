# torch 级实现: 直接用 ATen/native dropout（原生对照）。
from __future__ import annotations

import torch
import torch.nn.functional as F


def dropout_torch(input: torch.Tensor, p: float = 0.5,
                  train: bool = True) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("dropout expects a Tensor")
    return F.dropout(input, p, train)
