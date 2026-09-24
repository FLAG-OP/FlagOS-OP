# Triton 级实现: aten::clone 的 MLU 版本。
#
# 硬约束: 启动包 torch_device_fn.device（#11）；无 tl.sum，不涉 #15a；
# 不用 autotune（#15b）。纯拷贝，无算术。
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn  # #11 设备上下文

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_SUPPORTED = (torch.float16, torch.float32, torch.bfloat16)
_INT32_MAX = 2**31 - 1


def _fallback(src, memory_format):
    return torch.ops.aten.clone.default.redispatch(
        _FALLBACK_KEYSET, src, memory_format=memory_format
    )


@triton.jit
def _clone_contig(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, mask=m), mask=m)


@triton.jit
def _clone_1d(dst, src, n, ds0, ss0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i * ds0, tl.load(src + i * ss0, mask=m), mask=m)


@triton.jit
def _clone_2d(dst, src, n, s1, ds0, ds1, ss0, ss1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i1 = i % s1
    i0 = i // s1
    tl.store(dst + i0 * ds0 + i1 * ds1,
             tl.load(src + i0 * ss0 + i1 * ss1, mask=m), mask=m)


@triton.jit
def _clone_3d(dst, src, n, s1, s2, ds0, ds1, ds2, ss0, ss1, ss2,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i2 = i % s2
    t = i // s2
    i1 = t % s1
    i0 = t // s1
    doff = i0 * ds0 + i1 * ds1 + i2 * ds2
    soff = i0 * ss0 + i1 * ss1 + i2 * ss2
    tl.store(dst + doff, tl.load(src + soff, mask=m), mask=m)


@triton.jit
def _clone_4d(dst, src, n, s1, s2, s3, ds0, ds1, ds2, ds3, ss0, ss1, ss2, ss3,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i3 = i % s3
    t = i // s3
    i2 = t % s2
    t = t // s2
    i1 = t % s1
    i0 = t // s1
    doff = i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3
    soff = i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3
    tl.store(dst + doff, tl.load(src + soff, mask=m), mask=m)


def _collapse(shape, dstr, sstr):
    sizes, ds, ss = [], [], []
    for n, d, s in zip(shape, dstr, sstr):
        if n == 1:
            continue
        if sizes and ds[-1] == d * n and ss[-1] == s * n:
            sizes[-1] *= n
            ds[-1], ss[-1] = d, s
        else:
            sizes.append(n)
            ds.append(d)
            ss.append(s)
    return sizes, ds, ss


def _block(n):
    if n >= 65536:
        return 65536
    return max(1024, triton.next_power_of_2(n))


def _block_nd(n, rank):
    cap = 16384
    if n >= cap:
        return cap
    return max(1024, triton.next_power_of_2(n))


def _offsets_fit_int32(sizes, strides):
    return sum((k - 1) * abs(s) for k, s in zip(sizes, strides)) <= _INT32_MAX


def clone_triton(src: torch.Tensor,
                 memory_format=torch.preserve_format) -> torch.Tensor:
    """Triton-backed ``aten::clone`` for MLU（输出永不与 src 共享存储）。"""
    if not isinstance(src, torch.Tensor):
        raise TypeError("clone() expects a Tensor, got ", type(src))
    if memory_format is None:
        memory_format = torch.preserve_format

    if (
        src.layout != torch.strided
        or src.dtype not in _SUPPORTED
        or src._is_zerotensor()
        or src.is_neg()
    ):
        return _fallback(src, memory_format)

    n = src.numel()
    if n > _INT32_MAX:
        return _fallback(src, memory_format)

    out = torch.empty_like(src, memory_format=memory_format)
    if n == 0:
        return out

    with torch_device_fn.device(src.device):      # ← 硬约束 #11
        if out.is_contiguous() and src.is_contiguous():
            _clone_contig[(triton.cdiv(n, _block(n)),)](
                out, src, n, BLOCK=_block(n))
            return out

        sizes, ds, ss = _collapse(src.shape, out.stride(), src.stride())
        if not (_offsets_fit_int32(sizes, ds)
                and _offsets_fit_int32(sizes, ss)):
            out.copy_(src)
            return out

        r = len(sizes)
        B = _block_nd(n, r)
        g = (triton.cdiv(n, B),)
        if r == 1:
            _clone_1d[g](out, src, n, ds[0], ss[0], BLOCK=B)
        elif r == 2:
            _clone_2d[g](out, src, n, sizes[1], ds[0], ds[1], ss[0], ss[1],
                         BLOCK=B)
        elif r == 3:
            _clone_3d[g](out, src, n, sizes[1], sizes[2], *ds, *ss, BLOCK=B)
        elif r == 4:
            _clone_4d[g](out, src, n, sizes[1], sizes[2], sizes[3],
                         *ds, *ss, BLOCK=B)
        else:
            out.copy_(src)
    return out
