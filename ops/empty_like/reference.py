# empty_like 语义参考实现——判卷标准。
#
# 语义 (aten::empty_like(Tensor self, *, ScalarType? dtype=None,
#       Layout? layout=None, Device? device=None, bool? pin_memory=None,
#       MemoryFormat? memory_format=None) -> Tensor):
#   - 分配**未初始化**内存，shape 同 self；dtype/layout/device 默认继承 self。
#   - memory_format 默认 preserve_format（dense 保留 self.stride()，否则连续）。
#   - 只保证元数据（shape/dtype/stride/device），**不保证数值**（未初始化）。
from __future__ import annotations

import torch


def empty_like_reference(self: torch.Tensor, *, dtype=None, layout=None,
                         device=None, pin_memory=None, memory_format=None):
    if not isinstance(self, torch.Tensor):
        raise TypeError("empty_like expects a Tensor")
    return torch.empty_like(self, dtype=dtype, layout=layout, device=device,
                            pin_memory=pin_memory,
                            memory_format=memory_format)
