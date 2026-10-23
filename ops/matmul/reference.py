"""Independent CPU oracle for matmul forward and first-order gradients."""
from __future__ import annotations

import torch


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != b.dtype:
        raise RuntimeError("matmul inputs must have the same dtype")
    return torch.matmul(a.detach().cpu().float(),
                        b.detach().cpu().float()).to(a.dtype)


def matmul_backward_reference(a: torch.Tensor, b: torch.Tensor,
                              grad_output: torch.Tensor):
    left = a.detach().cpu().float().requires_grad_(True)
    right = b.detach().cpu().float().requires_grad_(True)
    result = torch.matmul(left, right)
    result.backward(grad_output.detach().cpu().float())
    return left.grad.to(a.dtype), right.grad.to(b.dtype)
