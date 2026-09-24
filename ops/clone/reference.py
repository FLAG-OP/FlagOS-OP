# clone 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor):
#   - 返回独立的**新**张量，shape/dtype/device 与 self 相同，绝不与 self
#     共享存储。
#   - memory_format 默认 preserve_format：non-overlapping 且 dense 的 self
#     保留 stride（contiguous/transposed），否则退化为 dense。
#   - 纯拷贝，无算术，故原生 clone 即位级判卷标准。
from __future__ import annotations

import torch


def clone_reference(src: torch.Tensor,
                    memory_format=torch.preserve_format) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError("clone expects a Tensor")
    if memory_format is None:
        memory_format = torch.preserve_format
    return src.clone(memory_format=memory_format)
