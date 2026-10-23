"""Triton matmul with runtime-specific workarounds and mathematical backward."""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._shape import matmul_backward, matrix_inputs, restore_output


@lru_cache(maxsize=1)
def _runtime_vendor() -> str:
    """Distinguish XMLIR's CUDA tensors from other vendors' CUDA tensors."""
    from flag_gems.runtime.backend.device import DeviceDetector

    return DeviceDetector().vendor_name


# Keep K in the runtime signature. P800's default BF16 lowering loses the
# masked K tail when K=1 is specialized: DMA loads one lane but MMA consumes
# 32, including stale SRAM. Dynamic K preserves the DS/MMA reduction mask.
# See reports/k1_root_cause.md for the reproducer and emitted IR evidence.
@triton.jit(do_not_specialize=["K"])
def _matmul_kernel(A, B, C, M, N, K,
                   sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    tiles_n = tl.cdiv(N, BN)
    rows = (tile // tiles_n) * BM + tl.arange(0, BM)
    cols = (tile % tiles_n) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for step in range(tl.cdiv(K, BK)):
        k = step * BK + ks
        a = tl.load(A + batch * sab + rows[:, None] * sam
                    + k[None, :] * sak,
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + batch * sbb + k[:, None] * sbk
                    + cols[None, :] * sbn,
                    mask=(k[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(a, b)

    tl.store(C + batch * scb + rows[:, None] * scm + cols[None, :] * scn,
             acc.to(C.dtype.element_ty),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    if _runtime_vendor() == "kunlunxin":
        # The locked P800 stack has produced wrong results for expanded matrix axes.
        if any(size > 1 and stride == 0 for size, stride in
               zip(left.shape[-2:], left.stride()[-2:])):
            left = left.contiguous()
        if any(size > 1 and stride == 0 for size, stride in
               zip(right.shape[-2:], right.stride()[-2:])):
            right = right.contiguous()
    bm = bn = 32
    bk = 32
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), batches)
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device(a.device):
        _matmul_kernel[grid](
            left, right, out, m, n, k,
            *left.stride(), *right.stride(), *out.stride(),
            BM=bm, BN=bn, BK=bk, num_warps=4,
        )
    return restore_output(out.reshape(out_shape), a_vector, b_vector)


def matmul_backward_triton(a: torch.Tensor, b: torch.Tensor,
                           grad_output: torch.Tensor,
                           need_a: bool = True, need_b: bool = True):
    """Compute first-order gradients, reducing broadcast batch dimensions."""
    return matmul_backward(a, b, grad_output, matmul_triton, need_a, need_b)
