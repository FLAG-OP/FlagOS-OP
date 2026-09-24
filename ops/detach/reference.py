# detach 语义参考实现。
#
# 语义 (aten::detach(Tensor(a) self) -> Tensor(a)):
#   返回与 self **共享底层存储**的新张量，从 autograd 图中脱离
#   (requires_grad=False)；不改变 self 的值与存储。
from __future__ import annotations

import torch


def detach_reference(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("detach expects a Tensor")
    return self.detach()
