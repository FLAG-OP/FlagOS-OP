# torch 级实现: 先按 stride 算所需存储，再 empty + as_strided。
#
# 不能调用 empty_strided 自身（递归）；用 torch.empty + as_strided 组合。
from __future__ import annotations

import torch


def _storage_needed(size, stride):
    """所需底层存储元素数: max offset + 1（非负 stride）。"""
    if any(s == 0 for s in size):
        return 0
    need = 1
    for s, k in zip(size, stride):
        need += (s - 1) * k
    return need


def empty_strided_torch(size, stride, *, dtype=None, layout=None,
                        device=None, pin_memory=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if layout is None:
        layout = torch.strided
    size = tuple(size)
    stride = tuple(stride)
    need = _storage_needed(size, stride)
    if need == 0:
        return torch.empty(size, dtype=dtype, layout=layout, device=device,
                           pin_memory=pin_memory)
    base = torch.empty(need, dtype=dtype, layout=layout, device=device,
                       pin_memory=pin_memory)
    return base.as_strided(size, stride)
