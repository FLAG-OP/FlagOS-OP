# torch 级实现: 不产生新设备码，用 ATen 组合表达 gelu。
#
# 用途:
#   - 作为 torch 级候选参与黄金/性能对照（与 reference 同语义）
#   - A1 注册的 fallback 实现（自研 Triton 不可用时直接委托）
#
# 递归规避（sdpa 教训, PLATFORM.md §3c）: A1 注册后 backend 内**不得**再调
# torch.nn.functional.gelu / torch.ops.aten.gelu（会重新进入本注册）。
# 这里用 redispatch 到 CompositeExplicitAutograd 落回原生实现。
from __future__ import annotations

import torch

# redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 后端键 CUDA
# 直下厂商 kernel——CompositeExplicitAutograd 落到组合分解, 半精度下
# 与厂商 kernel 数值分化（bf16 LN 差 0.031 同源实证）。
_FALLBACK_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)


def gelu_torch(self: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """与 reference 同语义的 ATen 组合（经 redispatch 防递归）。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("gelu expects a Tensor")
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    return torch.ops.aten.gelu.default.redispatch(
        _FALLBACK_KEYSET, self, approximate=approximate
    )
