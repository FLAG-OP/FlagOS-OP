# torch 级实现: 用 meta 推 stride + empty_strided 分配（不调用 empty 自身）。
from __future__ import annotations

import torch


def empty_torch(size, *, dtype=None, layout=None, device=None,
                pin_memory=None, memory_format=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if layout is None:
        layout = torch.strided
    if memory_format is None:
        memory_format = torch.contiguous_format
    shape = tuple(size)
    meta = torch.empty(shape, dtype=dtype, device="meta",
                       memory_format=memory_format)
    return torch.empty_strided(shape, meta.stride(), dtype=dtype,
                               layout=layout, device=device,
                               pin_memory=pin_memory)
