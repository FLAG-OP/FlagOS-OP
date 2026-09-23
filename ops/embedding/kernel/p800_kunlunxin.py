"""P800 Kunlunxin embedding backend.

The XMLIR stack exposes a highly optimized native row-gather through
``aten::index_select``.  The production backend deliberately delegates to it:
experiments in this repository showed that current Triton gathers are tens of
times slower (see reports/performance.md).  Dense backward delegates to
``aten::embedding_backward``; ``scale_grad_by_freq=True`` is implemented by
per-occurrence inverse-frequency scaling because the XPU native backward does
not implement that mode.
"""
from __future__ import annotations

import importlib.util

import torch


PLATFORM = "p800-kunlunxin"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # XMLIR presents XPU tensors as CUDA.
_HAS_KUNLUNXIN_STACK = None


def _has_kunlunxin_stack() -> bool:
    global _HAS_KUNLUNXIN_STACK
    if _HAS_KUNLUNXIN_STACK is None:
        if importlib.util.find_spec("torch_xmlir") is None:
            _HAS_KUNLUNXIN_STACK = False
        else:
            try:
                import torch_xmlir  # noqa: F401
                _HAS_KUNLUNXIN_STACK = True
            except Exception:
                _HAS_KUNLUNXIN_STACK = False
    return _HAS_KUNLUNXIN_STACK


def _validate_device_dtype(primary, indices, padding_idx, sparse,
                           num_weights):
    if primary.device.type not in SUPPORTED_DEVICE_TYPES or not _has_kunlunxin_stack():
        raise RuntimeError(
            f"embedding 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={primary.device!r} 或缺少 torch_xmlir。"
        )
    if primary.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported floating dtype: {primary.dtype}")
    if indices.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"unsupported indices dtype: {indices.dtype}")
    if primary.device != indices.device:
        raise ValueError("weight and indices must be on the same device")
    if sparse:
        raise NotImplementedError("sparse embedding is outside this delivery")
    if padding_idx is not None and padding_idx != -1:
        if padding_idx < 0 or padding_idx >= num_weights:
            raise IndexError(
                "padding_idx must be -1/None or inside [0, num_weights)"
            )


def _validate_forward(weight, indices, padding_idx, sparse):
    _validate_device_dtype(
        weight, indices, padding_idx, sparse, weight.shape[0]
    )
    if weight.dim() != 2:
        raise ValueError("weight must be 2D (num_weights, dim)")


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Native row-gather implementation of ``aten::embedding``."""
    # Hot path for the common 1D token-id call: one native row gather, no
    # Python reshape/view, and no semantic transform from embedding options.
    if (
        _has_kunlunxin_stack()
        and weight.dim() == 2
        and indices.dim() == 1
        and padding_idx in (-1, None)
        and not sparse
    ):
        return torch.ops.aten.index_select(weight, 0, indices)
    _validate_forward(weight, indices, padding_idx, sparse)
    # index_select requires a vector while aten::embedding accepts any index
    # shape.  Flatten here and restore the exact output shape.
    flat_indices = (
        indices if indices.dim() == 1 else indices.reshape(-1)
    )
    out = torch.ops.aten.index_select(weight, 0, flat_indices)
    if flat_indices is indices:
        return out
    return out.view(*indices.shape, weight.shape[-1])


def embedding_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Dense backward with a P800 fallback for inverse-frequency scaling."""
    _validate_device_dtype(
        grad_output, indices, padding_idx, sparse, num_weights
    )
    if num_weights < 0:
        raise ValueError("num_weights must be non-negative")
    if (grad_output.dim() != indices.dim() + 1
            or grad_output.shape[:-1] != indices.shape):
        raise ValueError("grad_output shape must be (*indices, dim)")

    if not scale_grad_by_freq:
        return torch.ops.aten.embedding_backward(
            grad_output, indices, num_weights, padding_idx,
            scale_grad_by_freq=False, sparse=False,
        )

    flat_indices = indices.reshape(-1)
    counts = torch.zeros(
        (num_weights,), device=indices.device, dtype=torch.int64
    )
    valid = flat_indices
    if padding_idx is not None and padding_idx >= 0:
        valid = flat_indices[flat_indices != padding_idx]
    if valid.numel():
        counts.index_add_(
            0, valid, torch.ones_like(valid, dtype=torch.int64)
        )
    inv_freq = counts.clamp_min(1).reciprocal()
    scaled = grad_output.float() * inv_freq[flat_indices].view(
        *indices.shape, 1
    )
    native = torch.ops.aten.embedding_backward(
        scaled.to(grad_output.dtype), indices, num_weights, padding_idx,
        scale_grad_by_freq=False, sparse=False,
    )
    return native.to(grad_output.dtype)
