# torch 级实现: 不产生新设备码，用 ATen 组合表达 type_as。
#
# 用途:
#   - 作为 torch 级候选参与黄金/性能对照（与 reference 同语义）
#   - A1 注册的 fallback 实现（自研 Triton 不可用时直接委托）
from __future__ import annotations

import torch


def type_as_torch(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """与 reference 同语义的 ATen 组合：``self.to(other.dtype)``。"""
    if not isinstance(self, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise TypeError("type_as expects two Tensors")
    if self.dtype == other.dtype:
        return self
    return self.to(other.dtype)
