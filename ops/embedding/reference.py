"""CPU/ATen semantic reference for ``aten::embedding``.

The reference deliberately uses index selection rather than
``F.embedding``/``aten::embedding`` so that candidate implementations are not
judged by the same dispatcher entry under test.
"""
from __future__ import annotations

import torch


def embedding_reference(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Return ``weight[index]`` for an arbitrary index tensor shape.

    ``padding_idx``, ``scale_grad_by_freq`` and ``sparse`` do not change the
    forward lookup.  They are retained to keep the public signature identical
    to ``aten::embedding``.
    """
    if weight.dim() != 2:
        raise ValueError("embedding weight must be 2D (num_weights, dim)")
    if indices.numel() and (indices.min().item() < 0
                            or indices.max().item() >= weight.shape[0]):
        raise IndexError("embedding indices are out of range")
    flat = indices.reshape(-1)
    out = weight.index_select(0, flat)
    return out.view(*indices.shape, weight.shape[-1])


def embedding_backward_reference(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Dense embedding gradient semantics used as the golden standard."""
    if sparse:
        raise NotImplementedError("this delivery covers dense backward only")
    if grad_output.shape != (*indices.shape, grad_output.shape[-1]):
        raise ValueError("grad_output shape must be (*indices, dim)")
    dim = grad_output.shape[-1]
    grad = torch.zeros(
        (num_weights, dim), device=grad_output.device,
        dtype=torch.float32,
    )
    flat_idx = indices.reshape(-1)
    flat_grad = grad_output.reshape(-1, dim).float()

    if scale_grad_by_freq:
        counts = torch.zeros(
            (num_weights,), device=indices.device, dtype=torch.int64
        )
        valid = flat_idx
        if padding_idx is not None and padding_idx >= 0:
            valid = flat_idx[flat_idx != padding_idx]
        counts.index_add_(
            0, valid, torch.ones_like(valid, dtype=torch.int64)
        )
        inv_freq = counts.clamp_min(1).reciprocal()
        flat_grad = flat_grad * inv_freq[flat_idx, None]

    for row, g in zip(flat_idx.tolist(), flat_grad.tolist()):
        if padding_idx is not None and row == padding_idx:
            continue
        grad[row] += torch.tensor(g, dtype=torch.float32)
    return grad.to(grad_output.dtype)
