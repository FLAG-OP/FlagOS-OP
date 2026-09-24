# Triton 级实现: aten::type_as 的 MLU 版本。
#
# 硬约束（本库实测）:
#   - 启动必须包 torch_device_fn.device 上下文（known-issues #11），
#     否则首次之后静默 no-op（输出未初始化内存，不报错）。
#   - 本算子为纯 elementwise copy/cast，无 tl.sum 归约，故 #15a
#     尾块归约坑不适用；masked load/store 本身安全。
#   - 不用 @triton.autotune（本栈会选出非法 num_warps，#15b）。
#
# 语义 (aten::type_as(self, other) -> Tensor): self cast 到 other.dtype，
# 留在 self.device；同 dtype 返回 self 本身。layout 决策交给
# empty_like(preserve_format)，与 ATen to() 一致。
#
# MLU 后端限制: 非 dense 的 self 在「窄 dtype -> 宽 dtype」
# （fp16/bf16 -> fp32）时，gather load + widening store 的当前
# Triton/MLU codegen 会崩，故该组合用 out.copy_(self) 兜底；dense/affine
# 路径（contiguous/transposed/permuted/channels_last）不受影响。
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


def _fallback(self: torch.Tensor, other: torch.Tensor):
    """原生 ATen type_as（autograd key 之下，不经 Triton）。"""
    return torch.ops.aten.type_as.default.redispatch(
        _FALLBACK_KEYSET, self, other
    )


@triton.jit
def _cast_contig(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(src + i, mask=m)
    tl.store(dst + i, v.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _cast_1d(dst, src, n, ds0, ss0, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(src + i * ss0, mask=m)
    tl.store(dst + i * ds0, v.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _cast_2d(dst, src, n, s1, ds0, ds1, ss0, ss1, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i1 = i % s1
    i0 = i // s1
    v = tl.load(src + i0 * ss0 + i1 * ss1, mask=m)
    tl.store(dst + i0 * ds0 + i1 * ds1, v.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _cast_3d(dst, src, n, s1, s2, ds0, ds1, ds2, ss0, ss1, ss2,
             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    i2 = i % s2
    t = i // s2
    i1 = t % s1
    i0 = t // s1
    v = tl.load(src + i0 * ss0 + i1 * ss1 + i2 * ss2, mask=m)
    tl.store(dst + i0 * ds0 + i1 * ds1 + i2 * ds2,
             v.to(dst.dtype.element_ty), mask=m)


@triton.jit
def _cast_4d(dst, src, n, s1, s2, s3, ds0, ds1, ds2, ds3, ss0, ss1, ss2, ss3,
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
    v = tl.load(src + i0 * ss0 + i1 * ss1 + i2 * ss2 + i3 * ss3, mask=m)
    tl.store(dst + i0 * ds0 + i1 * ds1 + i2 * ds2 + i3 * ds3,
             v.to(dst.dtype.element_ty), mask=m)


def _collapse(shape, dstr, sstr):
    """Drop size-1 dims and merge adjacent dims contiguous in both tensors."""
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
    """Block size for the strided (N-dim) kernels.

    The strided kernels materialise per-element index arrays in NRAM
    (16-20 B/elem); cap at 16384 so the worst case stays below the
    524288-byte NRAM limit.
    """
    cap = 16384
    if n >= cap:
        return cap
    return max(1024, triton.next_power_of_2(n))


def _offsets_fit_int32(sizes, strides):
    return sum((k - 1) * abs(s) for k, s in zip(sizes, strides)) <= _INT32_MAX


def type_as_triton(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Triton-backed ``aten::type_as`` for MLU.

    Returns ``self`` itself (aliased storage) when the dtypes already match;
    otherwise a new tensor cast to ``other.dtype`` on ``self.device``.
    """
    if not isinstance(self, torch.Tensor):
        raise TypeError("type_as() expects a Tensor as self, got ", type(self))
    if not isinstance(other, torch.Tensor):
        raise TypeError(
            "type_as(): argument 'other' (position 1) must be Tensor, not "
            + type(other).__name__
        )

    if self.dtype == other.dtype:
        return self

    target = other.dtype
    if (
        self.layout != torch.strided
        or self.dtype not in _SUPPORTED
        or target not in _SUPPORTED
        or self._is_zerotensor()
        or self.is_neg()
    ):
        return _fallback(self, other)

    n = self.numel()
    if n > _INT32_MAX:
        return _fallback(self, other)

    out = torch.empty_like(self, memory_format=torch.preserve_format,
                           dtype=target)
    if n == 0:
        return out

    with torch_device_fn.device(self.device):     # ← 硬约束 #11
        if out.stride() == self.stride():
            _cast_contig[(triton.cdiv(n, _block(n)),)](
                out, self, n, BLOCK=_block(n))
            return out

        if self.element_size() < out.element_size():
            # MLU codegen limitation: gather load of a narrower element type
            # followed by a widening store faults.  Delegate to copy_ into the
            # already-allocated (correctly typed/layed-out) out.
            out.copy_(self)
            return out

        sizes, ds, ss = _collapse(self.shape, out.stride(), self.stride())
        if not (_offsets_fit_int32(sizes, ds)
                and _offsets_fit_int32(sizes, ss)):
            out.copy_(self)
            return out

        r = len(sizes)
        B = _block_nd(n, r)
        g = (triton.cdiv(n, B),)
        if r == 1:
            _cast_1d[g](out, self, n, ds[0], ss[0], BLOCK=B)
        elif r == 2:
            _cast_2d[g](out, self, n, sizes[1], ds[0], ds[1], ss[0], ss[1],
                        BLOCK=B)
        elif r == 3:
            _cast_3d[g](out, self, n, sizes[1], sizes[2], *ds, *ss, BLOCK=B)
        elif r == 4:
            _cast_4d[g](out, self, n, sizes[1], sizes[2], sizes[3],
                        *ds, *ss, BLOCK=B)
        else:
            out.copy_(self)
    return out
