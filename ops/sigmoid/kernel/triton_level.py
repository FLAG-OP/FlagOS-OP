"""``aten::sigmoid`` 的可移植 Triton 实现。"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>

PLATFORM = "multi-triton"
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


def _native_fallback(x: torch.Tensor) -> torch.Tensor:
    """绕过当前 CUDA 注册，避免 A1 实现递归调用自身。"""
    return torch.ops.aten.sigmoid.default.redispatch(_FALLBACK_KEYSET, x)


def sigmoid_triton(x: torch.Tensor) -> torch.Tensor:
    """计算逐元素 sigmoid；fp16/bf16/fp32 使用 fp32 中间值。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"sigmoid expects a Tensor, got {type(x)!r}")
    if x.layout != torch.strided or x.dtype not in _SUPPORTED:
        return _native_fallback(x)

    try:
        from .backends import get_impl
    except ImportError:
        from kernel.backends import get_impl
    platform_impl = get_impl(x.device.type)
    if platform_impl is not None:
        return platform_impl(x)

    source = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(source)
    n_elements = source.numel()
    if n_elements == 0:
        return out

    # P800 的 Triton 后端适合较大的向量 program；与仓库已验证的
    # pointwise/clone 范式一致，大张量用 65536，避免数万个 program。
    block_size = (65536 if n_elements >= 65536
                  else max(256, triton.next_power_of_2(n_elements)))
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(source.device):
        _sigmoid_kernel[grid](source, out, n_elements, BLOCK_SIZE=block_size)
    return out
