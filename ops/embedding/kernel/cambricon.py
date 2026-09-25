"""Cambricon MLU590 embedding backend.

torch_mlu exposes an optimized native row-gather through ``aten::index_select``
and a *complete* native ``aten::embedding_backward``: unlike the XPU stack,
both ``padding_idx`` and ``scale_grad_by_freq=True`` are implemented and match
the CPU reference exactly (verified for 6 combinations in
reports/cambricon.md).  The production backend therefore delegates to both
native paths with no vendor-specific compensation -- the inverse-frequency
scaling that ``kernel/p800_kunlunxin.py`` needs on XPU is not required here.
"""
from __future__ import annotations

import importlib.util

import torch


PLATFORM = "cambricon"
SUPPORTED_DEVICE_TYPES = ("mlu",)  # torch_mlu presents MLU tensors as mlu.
_HAS_CAMBRICON_STACK = None


def _has_cambricon_stack() -> bool:
    global _HAS_CAMBRICON_STACK
    if _HAS_CAMBRICON_STACK is None:
        if importlib.util.find_spec("torch_mlu") is None:
            _HAS_CAMBRICON_STACK = False
        else:
            try:
                import torch_mlu  # noqa: F401
                _HAS_CAMBRICON_STACK = True
            except Exception:
                _HAS_CAMBRICON_STACK = False
    return _HAS_CAMBRICON_STACK


def _validate_device_dtype(primary, indices, padding_idx, num_weights):
    if primary.device.type not in SUPPORTED_DEVICE_TYPES or not _has_cambricon_stack():
        raise RuntimeError(
            f"embedding 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={primary.device!r} 或缺少 torch_mlu。"
        )
    if primary.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported floating dtype: {primary.dtype}")
    if indices.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"unsupported indices dtype: {indices.dtype}")
    if primary.device != indices.device:
        raise ValueError("weight and indices must be on the same device")
    if padding_idx is not None and padding_idx != -1:
        if padding_idx < 0 or padding_idx >= num_weights:
            raise IndexError(
                "padding_idx must be -1/None or inside [0, num_weights)"
            )


def _validate_forward(weight, indices, padding_idx, sparse):
    # ``sparse`` only requests a sparse weight gradient.  It does not change
    # the forward lookup, so sparse=True is accepted here and rejected only by
    # embedding_backward.
    _validate_device_dtype(weight, indices, padding_idx, weight.shape[0])
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
        _has_cambricon_stack()
        and weight.dim() == 2
        and indices.dim() == 1
        and padding_idx in (-1, None)
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
    """Dense backward delegating to the complete torch_mlu native kernel."""
    _validate_device_dtype(
        grad_output, indices, padding_idx, num_weights
    )
    if num_weights < 0:
        raise ValueError("num_weights must be non-negative")
    if sparse:
        # torch_mlu returns a sparse COO tensor for sparse=True instead of
        # rejecting it; this delivery covers dense backward only, so the
        # unsupported boundary is raised here (guard test relies on it).
        raise NotImplementedError(
            "sparse embedding backward is outside this delivery"
        )
    if (grad_output.dim() != indices.dim() + 1
            or grad_output.shape[:-1] != indices.shape):
        raise ValueError("grad_output shape must be (*indices, dim)")

    # torch_mlu implements padding_idx and scale_grad_by_freq natively and
    # matches the CPU reference exactly -- no inverse-frequency compensation
    # (the XPU-only path in p800_kunlunxin.py) is needed.
    return torch.ops.aten.embedding_backward(
        grad_output, indices, num_weights, padding_idx,
        scale_grad_by_freq=scale_grad_by_freq, sparse=False,
    )
