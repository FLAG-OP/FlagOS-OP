# torch 级实现: 用 ATen 组合表达 contiguous（含别名快路径）。
from __future__ import annotations

import torch


def contiguous_torch(inp: torch.Tensor,
                     memory_format=torch.contiguous_format) -> torch.Tensor:
    if not isinstance(inp, torch.Tensor):
        raise TypeError("contiguous expects a Tensor")
    if memory_format is None:
        memory_format = torch.contiguous_format
    if inp.layout == torch.strided and inp.is_contiguous(
            memory_format=memory_format):
        return inp
    out = torch.empty_like(inp, memory_format=memory_format)
    out.copy_(inp)
    return out
