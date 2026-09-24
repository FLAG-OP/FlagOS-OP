# detach_ 语义参考实现。
#
# 语义 (aten::detach_(Tensor(a!) self) -> Tensor(a!)):
#   原地把 self 从 autograd 图脱离 (requires_grad=False)，返回 self 本身。
from __future__ import annotations

import torch


def detach__reference(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("detach_ expects a Tensor")
    return self.detach_()
