"""Compatibility facade: route ``sdpa_triton`` by input device type.

Existing tests, golden scripts, and downstream code imported
``kernel.triton_level.sdpa_triton``.  Multi-platform support now lives in
``kernel.backends`` without changing that public path.
"""
from __future__ import annotations

from typing import Any

import torch

try:  # Package-style import: ops.sdpa.kernel.triton_level
    from .backends import get_impl
except ImportError:  # Standalone import with OP_DIR on sys.path
    from kernel.backends import get_impl


PLATFORM = "multi(ascend910,p800-kunlunxin,mlu590)"
SUPPORTED_DEVICE_TYPES = ("npu", "cuda", "mlu")


def sdpa_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch to the platform backend selected by ``query.device.type``."""
    return get_impl(query.device.type).sdpa_triton(
        query, key, value, attn_mask, dropout_p, is_causal, scale,
        enable_gqa, **kwargs,
    )
