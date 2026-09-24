# Triton 级实现: aten::dropout 的 MLU 版本（Philox 计数器 RNG）。
#
# 硬约束: 启动包 torch_device_fn.device（#11）；无归约，不涉 #15a；
# 不用 autotune（#15b）。RNG 流与原生不同 → 随机分支按结构/统计判定。
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
_BLOCK = 1024


def _fallback(input, p, train):
    return torch.ops.aten.dropout.default.redispatch(
        _FALLBACK_KEYSET, input, p, train
    )


def _fmt_float(p):
    return str(int(p)) if float(p).is_integer() else str(p)


def _suggest_memory_format(t):
    """Replicate ATen Tensor::suggest_memory_format for the strided case."""
    if t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if t.dim() == 5 and t.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    return torch.contiguous_format


def _philox_seed_offset(increment, device):
    """Read and advance the device Philox state（镜像 FlagGems）。"""
    idx = device.index
    if idx is None:
        idx = torch.mlu.current_device()
    gen = torch.mlu.default_generators[idx]
    state = gen.get_state()
    view = state.view(torch.int64)
    seed = int(view[0])
    offset = int(view[1])
    increment = (increment + 3) // 4 * 4
    view[1] = offset + increment
    gen.set_state(state)
    return seed, offset


@triton.jit(do_not_specialize=["p", "seed", "offset"])
def _dropout_kernel(X, Y, N, p, seed, offset, BLOCK: tl.constexpr):
    UNROLL: tl.constexpr = 4  # philox emits 128 random bits = 4 x uint32
    seed = seed.to(tl.int64)
    offset = offset.to(tl.int64)
    c0 = (offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, _O, _O)
    r0 = tl.uint_to_uniform_float(r0)
    r1 = tl.uint_to_uniform_float(r1)
    r2 = tl.uint_to_uniform_float(r2)
    r3 = tl.uint_to_uniform_float(r3)

    scale = 1.0 / (1.0 - p)
    base = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    o0, o1, o2, o3 = base, base + BLOCK, base + 2 * BLOCK, base + 3 * BLOCK

    x0 = tl.load(X + o0, mask=o0 < N, other=0.0)
    x1 = tl.load(X + o1, mask=o1 < N, other=0.0)
    x2 = tl.load(X + o2, mask=o2 < N, other=0.0)
    x3 = tl.load(X + o3, mask=o3 < N, other=0.0)

    y0 = tl.where(r0 > p, x0 * scale, 0.0)
    y1 = tl.where(r1 > p, x1 * scale, 0.0)
    y2 = tl.where(r2 > p, x2 * scale, 0.0)
    y3 = tl.where(r3 > p, x3 * scale, 0.0)

    tl.store(Y + o0, y0, mask=o0 < N)
    tl.store(Y + o1, y1, mask=o1 < N)
    tl.store(Y + o2, y2, mask=o2 < N)
    tl.store(Y + o3, y3, mask=o3 < N)


def dropout_triton(input: torch.Tensor, p: float = 0.5,
                   train: bool = True) -> torch.Tensor:
    """Triton-backed ``aten::dropout`` for MLU。"""
    if not isinstance(input, torch.Tensor):
        raise TypeError("dropout() expects a Tensor, got ", type(input))
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise RuntimeError(
            "dropout probability has to be between 0 and 1, but got "
            f"{_fmt_float(p)}"
        )

    if not train or p == 0.0:
        return input

    if (
        input.layout != torch.strided
        or input.dtype not in _SUPPORTED
        or input._is_zerotensor()
        or input.is_neg()
    ):
        return _fallback(input, p, train)

    fmt = _suggest_memory_format(input)
    if p == 1.0:
        return torch.zeros_like(input, memory_format=fmt)

    n = input.numel()
    if n > _INT32_MAX:
        return _fallback(input, p, train)

    if fmt == torch.contiguous_format:
        inp = input.contiguous()
        out = torch.empty_like(inp)
    elif input.is_contiguous(memory_format=fmt):
        inp = input
        out = torch.empty_like(input, memory_format=fmt)
    else:
        return _fallback(input, p, train)

    if n == 0:
        return out

    grid = (triton.cdiv(n, _BLOCK * 4),)
    with torch_device_fn.device(input.device):     # ← 硬约束 #11
        increment = grid[0] * _BLOCK
        seed, offset = _philox_seed_offset(increment, input.device)
        _dropout_kernel[grid](inp, out, n, p, seed, offset, BLOCK=_BLOCK)
    return out
