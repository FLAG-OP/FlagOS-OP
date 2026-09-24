# torch 级实现: 先搬到 CPU 再取标量（CPU item 为原生，避免自递归）。
from __future__ import annotations

import torch


def _local_scalar_dense_torch(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("_local_scalar_dense expects a Tensor")
    return self.cpu().item()
