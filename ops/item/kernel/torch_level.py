# torch 级实现: 先搬到 CPU 再取标量（CPU item 为原生，避免自递归）。
from __future__ import annotations

import torch


def item_torch(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("item expects a Tensor")
    return self.cpu().item()
