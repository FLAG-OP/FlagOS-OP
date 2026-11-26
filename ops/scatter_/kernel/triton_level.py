# Dashener2 2026-11-25-fix: stride-aware aten scatter_.src / scatter_.value.
# Dashener2 <dashen272@outlook.com>
import torch
import triton
import triton.language as tl

from ..validation import validate


@triton.jit
def _scatter(Dst, Index, Src, Value, N: tl.constexpr,
             SHAPE: tl.constexpr, DS: tl.constexpr, IS: tl.constexpr,
             SS: tl.constexpr, DIM: tl.constexpr, LIMIT: tl.constexpr,
             TENSOR_SRC: tl.constexpr, BLOCK: tl.constexpr, START: tl.constexpr = 0):
    linear = START + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < N
    rest = linear.to(tl.int64)
    io = tl.full((BLOCK,), 0, tl.int64)
    so = tl.full((BLOCK,), 0, tl.int64)
    do = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rest % SHAPE.value[d]
        rest = rest // SHAPE.value[d]
        io += coord * IS.value[d]
        if TENSOR_SRC:
            so += coord * SS.value[d]
        if d != DIM:
            do += coord * DS.value[d]
    idx = tl.load(Index + io, mask, other=0)
    valid = (idx >= 0) & (idx < LIMIT)
    tl.device_assert((~mask) | valid, "scatter_: index out of bounds")
    if TENSOR_SRC:
        value = tl.load(Src + so, mask, other=0)
    else:
        value = Value
    tl.store(Dst + do + idx * DS.value[DIM], value, mask & valid)


def scatter_triton(self, dim, index, src, *, _increment_version=True):
    """Inference scatter; src is a Tensor or scalar. Returns the same tensor.

    Duplicate tensor-source indices have an unspecified winning write, as in
    native scatter_. Bounds assertions are enabled even without TRITON_DEBUG.
    """
    dim = validate(self, dim, index, src)
    if index.numel() == 0:
        if _increment_version:
            torch.autograd.graph.increment_version(self)
        return self
    if self.device.type == "cpu":
        raise RuntimeError("scatter_triton requires an accelerator")
    from flag_gems.runtime import torch_device_fn

    # XMLIR 3.0 silently drops tl.device_assert. Validate before any writes,
    # including on backends which do support assertions, for a stable contract.
    if ((index < 0) | (index >= (self.shape or (1,))[dim])).any().item():
        raise RuntimeError("scatter_: index out of bounds")

    tensor_src = isinstance(src, torch.Tensor)
    if not tensor_src:
        # Native scalar conversion/overflow checks, without device allocation.
        src = torch.scalar_tensor(src, dtype=self.dtype).item()
    shape = tuple(index.shape) or (1,)
    ds = tuple(self.stride()) or (1,)
    ss = (tuple(src.stride()) or (1,)) if tensor_src else (0,) * len(shape)
    with torch_device_fn.device(self.device):
        args = (self, index, src if tensor_src else self, 0 if tensor_src else src,
                index.numel(), shape, ds, tuple(index.stride()) or (1,), ss,
                dim, (self.shape or (1,))[dim], tensor_src)
        block = 16384 if index.numel() % 16384 == 0 else 256
        full, tail = divmod(index.numel(), block)
        if full:
            _scatter[(full,)](*args, BLOCK=block, debug=True)
        # XMLIR masked scatter stores corrupt lane zero; repeated tail writes
        # also produced intermittent errors. Scalar tail programs avoid both.
        if tail:
            _scatter[(tail,)](*args, BLOCK=1, START=full * block, debug=True)
    if _increment_version:
        torch.autograd.graph.increment_version(self)
    return self
