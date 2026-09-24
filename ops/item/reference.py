# item 语义参考实现。
#
# 语义 (aten::item(Tensor self) -> Scalar):
#   单元素张量取为 Python 标量 (float/int/bool)；多元素抛 RuntimeError。
#   device->host 同步原语。
from __future__ import annotations

import torch


def item_reference(self):
    if not isinstance(self, torch.Tensor):
        raise TypeError("item expects a Tensor")
    return self.item()
