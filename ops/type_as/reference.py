# type_as 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::type_as(Tensor self, Tensor other) -> Tensor):
#   - 返回把 self cast 到 other.dtype 的张量，结果始终留在 self.device；
#     other 的 shape/strides/device 均被忽略，只取 dtype。
#   - self.dtype == other.dtype 时返回 self 本身（别名存储，不分配）。
#   - 其余 layout 决策沿用 to() 的 preserve_format：non-overlapping 且
#     dense 的 self 保留 stride（contiguous/transposed/channels_last），
#     overlapping 或 stepped（expand/strided slice）得到 dense 输出。
#   - cast 为 IEEE round-to-nearest，无算术放大，故 CPU 上原生 .to()
#     即为精确参考。golden 直接用它，不再叠加 fp32 往返（会把
#     bf16/fp16 输入先升精度而改变语义）。
#
# 不依赖任何注册/分发（纯函数，kernel 层/应用层共用）。
from __future__ import annotations

import torch


def type_as_reference(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """``self`` cast 到 ``other.dtype``，保持在 ``self.device``。"""
    if not isinstance(self, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise TypeError("type_as expects two Tensors")
    return self.to(other.dtype)
