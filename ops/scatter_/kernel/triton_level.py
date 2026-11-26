# Triton 级实现: aten::scatter_.src / scatter_.value（Ascend910，普通 Triton）。
# 参考 kernelGen 生成初稿的语义（TLE 版在本栈 TritonToLinalg 编译失败，
# 故改写为纯 tl.load/tl.store + 计算偏移；语义/边界与初稿一致）。
# Dashener2 2026-11-25-fix  # Dashener2 <dashen272@outlook.com>
from __future__ import annotations

import torch
import triton
import triton.language as tl

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd)
# dtypes the Ascend BiShengHIR path supports (float64 not supported -> native).
_TRITON_DTYPES = (
    torch.float16, torch.bfloat16, torch.float32,
    torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
)


def _fallback(self, dim, index, src):
    if isinstance(src, torch.Tensor):
        return torch.ops.aten.scatter_.src.redispatch(
            _FALLBACK_KEYSET, self, dim, index, src)
    return torch.ops.aten.scatter_.value.redispatch(
        _FALLBACK_KEYSET, self, dim, index, src)


@triton.jit
def _scatter_kernel(dst, index, src, val, base_off, src_base, stride_dim, n,
                    TENSOR_SRC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    idx = tl.load(index + i, mask=m, other=0).to(tl.int64)
    bo = tl.load(base_off + i, mask=m, other=0).to(tl.int64)
    if TENSOR_SRC:
        sbo = tl.load(src_base + i, mask=m, other=0).to(tl.int64)
        v = tl.load(src + sbo, mask=m, other=0)
    else:
        v = tl.load(val)
    tl.store(dst + bo + idx * stride_dim, v, mask=m)


def _load_validate():
    try:
        from ..validation import validate  # package mode
    except ImportError:
        from validation import validate  # script mode
    return validate


def _c_order_strides(shape):
    rank = len(shape)
    st = [1] * rank
    for d in range(rank - 2, -1, -1):
        st[d] = st[d + 1] * shape[d + 1]
    return st


def _all_offsets(index_shape, strides, n):
    rank = len(index_shape)
    cstride = _c_order_strides(index_shape)
    lin = torch.arange(n, dtype=torch.int64)
    off = torch.zeros(n, dtype=torch.int64)
    for d in range(rank):
        off += ((lin // cstride[d]) % index_shape[d]) * strides[d]
    return off


def _base_offsets(index_shape, self_strides, dim, n):
    """Non-scatter-dim contribution (host, shape/stride only; no index values)."""
    rank = len(index_shape)
    cstride = _c_order_strides(index_shape)
    lin = torch.arange(n, dtype=torch.int64)
    base = torch.zeros(n, dtype=torch.int64)
    for d in range(rank):
        if d == dim:
            continue
        coord = (lin // cstride[d]) % index_shape[d]
        base += coord * self_strides[d]
    return base


def scatter_triton(self, dim, index, src, *, _increment_version=True):
    """Inference scatter; ``src`` is a Tensor or scalar. Returns ``self``."""
    validate = _load_validate()
    dim = validate(self, dim, index, src)

    if index.numel() == 0:
        if _increment_version:
            torch.autograd.graph.increment_version(self)
        return self
    if self.device.type == "cpu":
        raise RuntimeError("scatter_triton requires an accelerator")
    if self.dtype not in _TRITON_DTYPES:
        return _fallback(self, dim, index, src)

    from flag_gems.runtime import torch_device_fn

    shape = tuple(self.shape) or (1,)
    limit = shape[dim]
    # Stable bounds contract (tl.device_assert not reliable on this backend).
    if ((index < 0) | (index >= limit)).any().item():
        raise RuntimeError("scatter_: index out of bounds")

    tensor_src = isinstance(src, torch.Tensor)
    index_flat = index.contiguous().reshape(-1)
    ishape = tuple(index.shape) or (1,)
    strides = tuple(self.stride()) or (1,)
    stride_dim = strides[dim]
    n = index_flat.numel()
    base = _base_offsets(ishape, strides, dim, n).to(self.device)

    if tensor_src:
        src_strides = tuple(src.stride()) or (1,)
        src_base = _all_offsets(ishape, src_strides, n).to(self.device)
        src_arg = src
        val = src_base  # unused
    else:
        src_base = base  # unused dummy
        src_arg = self  # unused dummy
        val = torch.tensor([src], dtype=self.dtype, device=self.device)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    with torch_device_fn.device(self.device):  # 硬约束 #11
        _scatter_kernel[grid](self, index_flat, src_arg, val, base, src_base,
                              stride_dim, n,
                              TENSOR_SRC=tensor_src, BLOCK=BLOCK)
    if _increment_version:
        torch.autograd.graph.increment_version(self)
    return self
