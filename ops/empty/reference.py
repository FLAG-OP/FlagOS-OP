# empty 语义参考实现——判卷标准（元数据）。
#
# 语义 (aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None,
#       Layout? layout=None, Device? device=None, bool? pin_memory=None,
#       MemoryFormat? memory_format=None) -> Tensor):
#   - 分配 size 形状的**未初始化**张量；仅保证元数据，数值不保证。
from __future__ import annotations

import torch


def empty_reference(size, *, dtype=None, layout=None, device=None,
                    pin_memory=None, memory_format=None):
    return torch.empty(size, dtype=dtype, layout=layout, device=device,
                       pin_memory=pin_memory, memory_format=memory_format)
