# contiguous 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor):
#   - 返回在 memory_format 下连续的张量。
#   - 若 self 已在该 format 下连续，返回 **self 本身**（别名存储，不拷贝）。
#   - 否则分配新张量并搬运。纯拷贝，无算术，原生即精确判卷标准。
from __future__ import annotations

import torch


def contiguous_reference(inp: torch.Tensor,
                         memory_format=torch.contiguous_format
                         ) -> torch.Tensor:
    if not isinstance(inp, torch.Tensor):
        raise TypeError("contiguous expects a Tensor")
    if memory_format is None:
        memory_format = torch.contiguous_format
    return inp.contiguous(memory_format=memory_format)
