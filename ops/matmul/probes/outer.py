"""Diagnostic matmul without tl.dot; not used by the production/A1 path."""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ops.matmul.kernel._shape import matrix_inputs, restore_output


@triton.jit
def _matmul_outer_kernel(A, B, C,
                         sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    rows = (tile // tiles_n) * BM + tl.arange(0, BM)
    cols = (tile % tiles_n) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(K):
        a = tl.load(A + batch * sab + rows * sam + k * sak,
                    mask=rows < M, other=0).to(tl.float32)
        b = tl.load(B + batch * sbb + k * sbk + cols * sbn,
                    mask=cols < N, other=0).to(tl.float32)
        acc += a[:, None] * b[None, :]

    tl.store(C + batch * scb + rows[:, None] * scm + cols[None, :] * scn,
             acc.to(C.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _matmul_vector_kernel(A, B, C, M, N, K,
                          sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                          BK: tl.constexpr):
    index = tl.program_id(0)
    batch = tl.program_id(1)
    row = index // N
    col = index % N
    ks = tl.arange(0, BK)
    a = tl.load(A + batch * sab + row * sam + ks * sak,
                mask=ks < K, other=0).to(tl.float32)
    b = tl.load(B + batch * sbb + ks * sbk + col * sbn,
                mask=ks < K, other=0).to(tl.float32)
    result = tl.sum(a * b, 0)
    tl.store(C + batch * scb + row * scm + col * scn,
             result.to(C.dtype.element_ty))


def matmul_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Explicit forward-only diagnostic entry, including vector/batch shaping."""
    if a.device != b.device or a.dtype != b.dtype:
        raise RuntimeError("matmul inputs must have the same device and dtype")
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(f"matmul supports float32, float16, bfloat16; got {a.dtype}")

    left, right, batch_shape, a_vector, b_vector = matrix_inputs(a, b)
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    batches = math.prod(batch_shape)
    out_shape = batch_shape + (m, n)
    if batches == 0 or m == 0 or n == 0:
        return restore_output(torch.empty(out_shape, device=a.device,
                                          dtype=a.dtype), a_vector, b_vector)
    if k == 0:
        return restore_output(torch.zeros(out_shape, device=a.device,
                                          dtype=a.dtype), a_vector, b_vector)

    left = left.reshape(batches, m, k)
    right = right.reshape(batches, k, n)
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device(a.device):
        if m <= 2 or n <= 2:
            _matmul_vector_kernel[(m * n, batches)](
                left, right, out, m, n, k,
                *left.stride(), *right.stride(), *out.stride(),
                BK=triton.next_power_of_2(k), num_warps=4,
            )
        else:
            bm = bn = 32
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), batches)
            _matmul_outer_kernel[grid](
                left, right, out,
                *left.stride(), *right.stride(), *out.stride(),
                M=m, N=n, K=k, BM=bm, BN=bn, num_warps=4,
            )
    return restore_output(out.reshape(out_shape), a_vector, b_vector)
