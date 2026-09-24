# torch 级实现: 直接用 ATen copy_（原生对照）。
from __future__ import annotations

import torch


def copy_torch(dst: torch.Tensor, src: torch.Tensor,
               non_blocking: bool = False) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        if isinstance(src, (int, float, bool)):
            return dst.fill_(src)
        raise TypeError("unsupported src type for copy_: ", type(src))
    return dst.copy_(src, non_blocking=non_blocking)
