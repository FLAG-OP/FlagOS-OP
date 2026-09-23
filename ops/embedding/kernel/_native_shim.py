"""Native ``aten::embedding`` comparison shim."""
from __future__ import annotations

import torch


def embedding_native(weight, indices, padding_idx=-1,
                     scale_grad_by_freq=False, sparse=False):
    return torch.ops.aten.embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )
