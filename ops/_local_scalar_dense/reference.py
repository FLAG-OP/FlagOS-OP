# _local_scalar_dense 语义参考实现。
#
# 语义 (aten::_local_scalar_dense(Tensor self) -> Scalar):
#   单元素稠密张量取为 host 标量；item() 的底层原语。
from __future__ import annotations

import torch


def _local_scalar_dense_reference(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("_local_scalar_dense expects a Tensor")
    return self.item()
