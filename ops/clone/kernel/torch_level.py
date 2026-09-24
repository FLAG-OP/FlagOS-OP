# torch 级实现: 用 ATen 组合表达 clone（不产生新设备码）。
#
# 与 reference 的差异: reference 直接用原生 `self.clone`；torch 级显式
# 走 `empty_like(preserve_format)` 决定 layout + `copy_` 搬数据，便于
# 与 Triton 级对照（同样的 layout 决策，不同的搬运实现）。
from __future__ import annotations

import torch


def clone_torch(src: torch.Tensor,
                memory_format=torch.preserve_format) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError("clone expects a Tensor")
    if memory_format is None:
        memory_format = torch.preserve_format
    out = torch.empty_like(src, memory_format=memory_format)
    out.copy_(src)
    return out
