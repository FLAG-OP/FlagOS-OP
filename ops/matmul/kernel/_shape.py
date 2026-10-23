"""Shared torch.matmul shape handling for the reference and device paths."""
from __future__ import annotations

import torch


def matrix_inputs(a: torch.Tensor, b: torch.Tensor):
    if a.ndim == 0 or b.ndim == 0:
        raise RuntimeError("matmul requires both arguments to be at least 1-D")
    a_vector, b_vector = a.ndim == 1, b.ndim == 1
    left = a.unsqueeze(0) if a_vector else a
    right = b.unsqueeze(-1) if b_vector else b
    if left.shape[-1] != right.shape[-2]:
        raise RuntimeError(
            f"matmul contraction dimensions differ: {left.shape[-1]} "
            f"and {right.shape[-2]}"
        )
    batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = left.expand(batch_shape + left.shape[-2:])
    right = right.expand(batch_shape + right.shape[-2:])
    return left, right, batch_shape, a_vector, b_vector


def restore_output(out: torch.Tensor, a_vector: bool, b_vector: bool):
    if a_vector:
        out = out.squeeze(-2)
    if b_vector:
        out = out.squeeze(-1)
    return out


def matmul_backward(a: torch.Tensor, b: torch.Tensor,
                    grad_output: torch.Tensor, matmul,
                    need_a: bool = True, need_b: bool = True):
    """Apply the same broadcast and vector gradient rules to either backend."""
    a_vector, b_vector = a.ndim == 1, b.ndim == 1
    left = a.unsqueeze(0) if a_vector else a
    right = b.unsqueeze(-1) if b_vector else b
    grad = grad_output
    if b_vector:
        grad = grad.unsqueeze(-1)
    if a_vector:
        grad = grad.unsqueeze(-2)

    grad_a = grad_b = None
    if need_a:
        grad_a = matmul(grad, right.transpose(-2, -1))
        grad_a = grad_a.sum_to_size(*left.shape)
        if a_vector:
            grad_a = grad_a.squeeze(-2)
    if need_b:
        grad_b = matmul(left.transpose(-2, -1), grad)
        grad_b = grad_b.sum_to_size(*right.shape)
        if b_vector:
            grad_b = grad_b.squeeze(-1)
    return grad_a, grad_b
