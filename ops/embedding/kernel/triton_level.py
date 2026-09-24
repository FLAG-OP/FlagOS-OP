"""Experimental deterministic Triton gather (not the production backend).

The kernel is retained as a reproducible platform probe.  It is correct for
ordinary dense lookup, but XMLIR lowers this gather tens of times slower than
the native row-gather.  Production callers should use
``kernel.p800_kunlunxin.embedding``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit
def _embedding_gather_kernel(
    OUT, INDICES, WEIGHT, elem_count, dim,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elem_count
    rows = offsets // dim
    cols = offsets % dim
    row_idx = tl.load(INDICES + rows, mask=mask, other=0)
    values = tl.load(
        WEIGHT + row_idx * dim + cols, mask=mask, other=0.0
    )
    tl.store(OUT + offsets, values, mask=mask)


def embedding_triton(weight, indices, padding_idx=-1,
                     scale_grad_by_freq=False, sparse=False):
    # sparse only selects a sparse backward representation; forward is unchanged.
    dim = weight.shape[-1]
    flat_indices = indices.reshape(-1).contiguous()
    weight = weight.contiguous()
    elem_count = flat_indices.numel() * dim
    out = torch.empty(
        (elem_count,), device=weight.device, dtype=weight.dtype
    ).view(*indices.shape, dim)
    block = 1024
    grid = (triton.cdiv(elem_count, block),)
    with torch_device_fn.device(weight.device):
        _embedding_gather_kernel[grid](
            out.reshape(-1), flat_indices, weight, elem_count, dim,
            BLOCK=block, num_warps=4, num_stages=1,
        )
    return out
