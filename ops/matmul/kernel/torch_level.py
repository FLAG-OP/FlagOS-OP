"""ATen composition baseline; this path does not implement a new kernel."""
from __future__ import annotations

import math

import torch

from ._shape import matmul_backward, matrix_inputs, restore_output


def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    left, right, batch_shape, a_vector, b_vector = matrix_inputs(a, b)
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    batches = math.prod(batch_shape)
    if batches == 0:
        out = torch.empty(batch_shape + (m, n), device=a.device, dtype=a.dtype)
        out = out + left.sum() * 0 + right.sum() * 0
        return restore_output(out, a_vector, b_vector)
    out = torch.bmm(
        left.reshape(batches, m, k), right.reshape(batches, k, n)
    ).reshape(batch_shape + (m, n))
    return restore_output(out, a_vector, b_vector)


def matmul_backward_torch(a: torch.Tensor, b: torch.Tensor,
                          grad_output: torch.Tensor,
                          need_a: bool = True, need_b: bool = True):
    return matmul_backward(a, b, grad_output, matmul_torch, need_a, need_b)
