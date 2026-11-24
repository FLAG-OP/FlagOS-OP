"""ATen composition baseline; this path does not implement a new kernel."""
from __future__ import annotations

import math

import torch

from ._shape import matmul_backward, matrix_inputs, restore_output


def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device.type == "supa":
        # The Biren JIT imports this SDK module in its generated process.
        # Importing it in the parent initializes the vendor runtime for all
        # supported matmul shapes.
        import br_generator  # noqa: F401

    left, right, batch_shape, a_vector, b_vector = matrix_inputs(a, b)
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    batches = math.prod(batch_shape)
    if batches == 0:
        out = torch.empty(batch_shape + (m, n), device=a.device, dtype=a.dtype)
        out = out + left.sum() * 0 + right.sum() * 0
        return restore_output(out, a_vector, b_vector)
    if batches == 1 and left.ndim == right.ndim == 2:
        if left.device.type == "supa":
            # Biren's generic mm/bmm autograd path requests a missing
            # br_generator. Use the vendor matmul primitive explicitly; the
            # surrounding A1 autograd.Function supplies the mathematical
            # first-order backward.
            out = torch.empty((m, n), device=left.device, dtype=left.dtype)
            out = torch.ops.supa.br_matmul_with_output(
                left, right, False, False, out
            )
        else:
            out = torch.mm(left, right)
        return restore_output(out.reshape(batch_shape + (m, n)),
                              a_vector, b_vector)
    left_batch = left.reshape(batches, m, k)
    right_batch = right.reshape(batches, k, n)
    if left.device.type == "supa":
        # Biren's packed bmm path is unreliable for broadcast BF16 batches.
        # Correctness comes first on this platform; use the vendor 2-D mm
        # primitive once per batch and leave packed bmm for other vendors.
        left_batch = left_batch.contiguous()
        right_batch = right_batch.contiguous()
        out_batch = torch.empty((batches, m, n),
                                device=left.device, dtype=left.dtype)
        for batch_index in range(batches):
            out_batch[batch_index] = torch.ops.supa.br_matmul_with_output(
                left_batch[batch_index], right_batch[batch_index],
                False, False, out_batch[batch_index]
            )
        out = out_batch
    else:
        out = torch.bmm(left_batch, right_batch)
    out = out.reshape(batch_shape + (m, n))
    return restore_output(out, a_vector, b_vector)


def matmul_backward_torch(a: torch.Tensor, b: torch.Tensor,
                          grad_output: torch.Tensor,
                          need_a: bool = True, need_b: bool = True):
    return matmul_backward(a, b, grad_output, matmul_torch, need_a, need_b)
