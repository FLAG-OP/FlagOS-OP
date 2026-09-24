# Triton 级实现: aten::copy_ 的 MLU 版本（含 2D 转置型分块加速）。
#
# 硬约束: 启动包 torch_device_fn.device（#11）；无归约，不涉 #15a；
# 不用 autotune（#15b）。原地写 dst 并返回 dst；src 支持广播。
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


def _fallback(dst, src, non_blocking):
    return torch.ops.aten.copy_.default.redispatch(
        _FALLBACK_KEYSET, dst, src, non_blocking
    )


@triton.jit
def _copy_1d(dst, src, n, ds0, ss0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i * ds0, tl.load(src + i * ss0, mask=m), mask=m)


@triton.jit
def _copy_contig(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(dst + i, tl.load(src + i, mask=m), mask=m)


@triton.jit
def _copy_2d(dst, src, n, s1, ds0, ds1, ss0, ss1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i1 = i % s1
    i0 = i // s1
    tl.store(dst + i0 * ds0 + i1 * ds1,
             tl.load(src + i0 * ss0 + i1 * ss1, mask=m), mask=m)


@triton.jit
def _copy_2d_swiz(dst, src, M, N, sc, dr, BM: tl.constexpr, BN: tl.constexpr):
    """Transpose-like copy: src fast on dim0, dst fast on dim1 (or mirror)."""
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
def _copy_3d(dst, src, n, s1, s2, ds0, ds1, ds2, ss0, ss1, ss2,
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
def _copy_4d(dst, src, n, s1, s2, s3, ds0, ds1, ds2, ds3, ss0, ss1, ss2, ss3,
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


def _swiz_block(M, N, elem_size):
    cap = 65536 if elem_size <= 2 else 32768
    BM = min(1024, triton.next_power_of_2(M))
    BN = min(cap // BM, triton.next_power_of_2(N))
    if BN < 1:
        BN = 1
    return BM, BN


def copy_triton(dst: torch.Tensor, src: torch.Tensor,
                non_blocking: bool = False) -> torch.Tensor:
    """Triton-backed ``aten::copy_`` for MLU（原地写 dst，返回 dst）。"""
    if not isinstance(src, torch.Tensor):
        if isinstance(src, (int, float, bool)):
            return dst.fill_(src)
        raise TypeError("unsupport src type for copy_: ", type(src))
    if (
        dst.device != src.device
        or dst.dtype not in _SUPPORTED
        or src.dtype not in _SUPPORTED
        or dst.layout != torch.strided
        or src.layout != torch.strided
        or dst._is_zerotensor()
        or src._is_zerotensor()
        or dst.is_neg() or src.is_neg()
    ):
        return _fallback(dst, src, non_blocking)
    n = dst.numel()
    if n == 0 or n > _INT32_MAX or torch._C._is_alias_of(dst, src):
        return _fallback(dst, src, non_blocking)

    if src.shape != dst.shape:
        if torch.broadcast_shapes(dst.shape, src.shape) != dst.shape:
            raise RuntimeError(
                f"The broadcast shape does not match destination shape "
                f"{tuple(dst.shape)}")
        src = src.expand(dst.shape)

    with torch_device_fn.device(dst.device):      # ← 硬约束 #11
        if dst.is_contiguous() and src.is_contiguous():
            _copy_contig[(triton.cdiv(n, _block(n)),)](
                dst, src, n, BLOCK=_block(n))
            return dst

        sizes, ds, ss = _collapse(dst.shape, dst.stride(), src.stride())
        if (sum((k - 1) * abs(d) for k, d in zip(sizes, ds)) > _INT32_MAX
                or sum((k - 1) * abs(s) for k, s in zip(sizes, ss))
                > _INT32_MAX):
            return _fallback(dst, src, non_blocking)

        r = len(sizes)
        B = _block_nd(n, r)
        g = (triton.cdiv(n, B),)
        if r == 1:
            _copy_1d[g](dst, src, n, ds[0], ss[0], BLOCK=B)
        elif r == 2 and ss[0] == 1 and ds[1] == 1:
            BM, BN = _swiz_block(sizes[0], sizes[1], dst.element_size())
            _copy_2d_swiz[(triton.cdiv(sizes[0], BM)
                           * triton.cdiv(sizes[1], BN),)](
                dst, src, sizes[0], sizes[1], ss[1], ds[0], BM=BM, BN=BN)
        elif r == 2 and ss[1] == 1 and ds[0] == 1:
            BM, BN = _swiz_block(sizes[1], sizes[0], dst.element_size())
            _copy_2d_swiz[(triton.cdiv(sizes[1], BM)
                           * triton.cdiv(sizes[0], BN),)](
                dst, src, sizes[1], sizes[0], ss[0], ds[1], BM=BM, BN=BN)
        elif r == 2:
            _copy_2d[g](dst, src, n, sizes[1], ds[0], ds[1], ss[0], ss[1],
                        BLOCK=B)
        elif r == 3:
            _copy_3d[g](dst, src, n, sizes[1], sizes[2], *ds, *ss, BLOCK=B)
        elif r == 4:
            _copy_4d[g](dst, src, n, sizes[1], sizes[2], sizes[3],
                        *ds, *ss, BLOCK=B)
        else:
            return _fallback(dst, src, non_blocking)
    return dst
