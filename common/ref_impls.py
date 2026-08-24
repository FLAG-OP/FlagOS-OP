# 共享 PyTorch 参考实现。
# 开发流程: 先写参考实现确定语义，再让 Triton/vendor 实现对齐它。
from __future__ import annotations

import torch


def gelu_and_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """GELU(tanh 近似) * gate，vLLM GeluAndMul 语义。"""
    xf = x.float()
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    gelu = 0.5 * xf * (1.0 + torch.tanh(inner))
    return (gelu * gate.float()).to(x.dtype)
