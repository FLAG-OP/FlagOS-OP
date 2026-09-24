# Triton 级实现: aten::contiguous 的 MLU 版本（contiguous_format 路径）。
#
# 硬约束: 启动包 torch_device_fn.device（#11）；无归约，不涉 #15a；
# 不用 autotune（#15b）。已连续时返回 self 本身（别名）。
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


def _fallback(inp, memory_format):
    return torch.ops.aten.contiguous.default.redispatch(
        _FALLBACK_KEYSET, inp, memory_format=memory_format
    )


@triton.jit
def _contig_copy(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, mask=m), mask=m)


@triton.jit
def _contig_1d(dst, src, n, ds0, ss0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i * ds0, tl.load(src + i * ss0, mask=m), mask=m)


@triton.jit
def _contig_2d_swiz(dst, src, M, N, sc, dr, BM: tl.constexpr, BN: tl.constexpr):
    """Transpose-like copy: src fast on dim0, dst fast on dim1 (or mirror).

    Tile V[j, i] = src[r0+j, c0+i]，两个最内维 (j) 在 load/store 都连续，
    后端走 burst 访问而非逐元素 gather。
    """
    pid = tl.program_id(0)
    ncb = tl.cdiv(N, BN)
    ta = pid // ncb
    tb = pid % ncb
    r0 = ta * BM + tl.arange(0, BM)
    c0 = tb * BN + tl.arange(0, BN)
    rm = r0[None, :] < M
    cm = c0[:, None] < N
    val = tl.load(src + r0[None, :] + c0[:, None] * sc, mask=rm & cm)
    tl.store(dst + r0[None, :] * dr + c0[:, None], val, mask=rm & cm)


@triton.jit
def _contig_2d(dst, src, n, s1, ds0, ds1, ss0, ss1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i1 = i % s1
    i0 = i // s1
    tl.store(dst + i0 * ds0 + i1 * ds1,
             tl.load(src + i0 * ss0 + i1 * ss1, mask=m), mask=m)


@triton.jit
def _contig_3d(dst, src, n, s1, s2, ds0, ds1, ds2, ss0, ss1, ss2,
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
def _contig_4d(dst, src, n, s1, s2, s3, ds0, ds1, ds2, ds3, ss0, ss1, ss2, ss3,
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


def _swiz_block(M, N, elem_size):
    """Tile (BM x BN) for _contig_2d_swiz；NRAM 预算与 copy_r3 一致。"""
    cap = 65536 if elem_size <= 2 else 32768
    BM = min(1024, triton.next_power_of_2(M))
    BN = min(cap // BM, triton.next_power_of_2(N))
    if BN < 1:
        BN = 1
    return BM, BN


def contiguous_triton(inp: torch.Tensor,
                      memory_format=torch.contiguous_format) -> torch.Tensor:
    """Triton-backed ``aten::contiguous`` for MLU。"""
    if not isinstance(inp, torch.Tensor):
        raise TypeError("contiguous() expects a Tensor, got ", type(inp))
    if memory_format is None:
        memory_format = torch.contiguous_format

    # 别名快路径: 已连续则返回 self 本身
    if inp.layout == torch.strided and inp.is_contiguous(
            memory_format=memory_format):
        return inp

    if (
        memory_format != torch.contiguous_format
        or inp.layout != torch.strided
        or inp.dtype not in _SUPPORTED
        or inp._is_zerotensor()
        or inp.is_neg()
    ):
        return _fallback(inp, memory_format)

    n = inp.numel()
    if n > _INT32_MAX:
        return _fallback(inp, memory_format)

    out = torch.empty_like(inp, memory_format=memory_format)
    if n == 0:
        return out

    sizes, ds, ss = _collapse(inp.shape, out.stride(), inp.stride())
    if not (_offsets_fit_int32(sizes, ds)
            and _offsets_fit_int32(sizes, ss)):
        out.copy_(inp)
        return out

    r = len(sizes)
    B = _block_nd(n, r)
    g = (triton.cdiv(n, B),)
    with torch_device_fn.device(inp.device):      # ← 硬约束 #11
        if r == 1:
            _contig_1d[g](out, inp, n, ds[0], ss[0], BLOCK=B)
        elif r == 2 and ss[0] == 1 and ds[1] == 1:
            # 转置型: src 快维在 dim0、dst 快维在 dim1 → 分块 burst 拷贝
            BM, BN = _swiz_block(sizes[0], sizes[1], out.element_size())
            _contig_2d_swiz[(triton.cdiv(sizes[0], BM)
                             * triton.cdiv(sizes[1], BN),)](
                out, inp, sizes[0], sizes[1], ss[1], ds[0], BM=BM, BN=BN)
        elif r == 2 and ss[1] == 1 and ds[0] == 1:
            BM, BN = _swiz_block(sizes[1], sizes[0], out.element_size())
            _contig_2d_swiz[(triton.cdiv(sizes[1], BM)
                             * triton.cdiv(sizes[0], BN),)](
                out, inp, sizes[1], sizes[0], ss[0], ds[1], BM=BM, BN=BN)
        elif r == 2:
            _contig_2d[g](out, inp, n, sizes[1], ds[0], ds[1], ss[0], ss[1],
                          BLOCK=B)
        elif r == 3:
            _contig_3d[g](out, inp, n, sizes[1], sizes[2], *ds, *ss, BLOCK=B)
        elif r == 4:
            _contig_4d[g](out, inp, n, sizes[1], sizes[2], sizes[3],
                          *ds, *ss, BLOCK=B)
        else:
            out.copy_(inp)
    return out
