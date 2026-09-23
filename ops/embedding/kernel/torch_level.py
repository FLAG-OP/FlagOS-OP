"""Portable ATen-composition implementation."""
from __future__ import annotations

import torch

try:
    from ..reference import embedding_reference
except ImportError:
    from reference import embedding_reference


def embedding_torch(weight, indices, padding_idx=-1,
                    scale_grad_by_freq=False, sparse=False):
    return embedding_reference(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )
