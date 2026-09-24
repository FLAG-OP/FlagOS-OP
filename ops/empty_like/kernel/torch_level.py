# torch 级实现: 用 empty_strided 组合出 empty_like 的元数据语义。
#
# 未初始化 tensor 没有可编译的设备码，故本算子无 Triton 级（见 kernel/README.md）。
# 需要先算出目标 layout 的 stride（用 meta 设备推导），再 empty_strided 分配。
from __future__ import annotations

import torch


def _target_strides(self, memory_format):
    if memory_format == torch.preserve_format:
        return tuple(self.to("meta").stride())
    shape = tuple(self.shape)
    return tuple(torch.empty(shape, dtype=self.dtype, device="meta",
                             memory_format=memory_format).stride())


def empty_like_torch(self: torch.Tensor, *, dtype=None, layout=None,
                     device=None, pin_memory=None, memory_format=None):
    if not isinstance(self, torch.Tensor):
        raise TypeError("empty_like expects a Tensor")
    dtype = self.dtype if dtype is None else dtype
    layout = self.layout if layout is None else layout
    device = self.device if device is None else device
    if memory_format is None:
        memory_format = torch.preserve_format
    strides = _target_strides(self, memory_format)
    return torch.empty_strided(tuple(self.shape), strides, dtype=dtype,
                               layout=layout, device=device,
                               pin_memory=pin_memory)
