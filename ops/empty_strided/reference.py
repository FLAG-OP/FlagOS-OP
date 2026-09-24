# empty_strided 语义参考实现——判卷标准（元数据）。
#
# 语义 (aten::empty_strided(SymInt[] size, SymInt[] stride, *,
#       ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#       bool? pin_memory=None) -> Tensor):
#   - 按指定 stride 分配**未初始化**张量；允许非连续/重叠 stride。
#   - 只保证元数据（shape/dtype/stride/device），数值不保证。
from __future__ import annotations

import torch


def empty_strided_reference(size, stride, *, dtype=None, layout=None,
                            device=None, pin_memory=None):
    return torch.empty_strided(size, stride, dtype=dtype, layout=layout,
                               device=device, pin_memory=pin_memory)
