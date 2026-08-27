from __future__ import annotations

import torch


def ref(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    gelu = 0.5 * xf * (1.0 + torch.tanh(inner))
    return (gelu * gate.float()).to(x.dtype)
