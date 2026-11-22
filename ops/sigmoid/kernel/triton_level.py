"""``aten::sigmoid`` 的可移植 Triton 实现。"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>
# wt-2026-11-20-fix
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


@triton.jit
def _sigmoid_255_no_mask_kernel(x_ptr, y_ptr):
    """P800/XMLIR 的 255 元素无 mask 确定性路径。"""
    offsets = tl.arange(0, 256)
    # 第 256 个 lane 重复读取最后一个合法元素；所有访存均无 mask。
    safe_offsets = tl.minimum(offsets, 254)
    x = tl.load(x_ptr + safe_offsets).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y)


def _native_fallback(x: torch.Tensor) -> torch.Tensor:
    """绕过当前 CUDA 注册，避免 A1 实现递归调用自身。"""
    return torch.ops.aten.sigmoid.default.redispatch(_FALLBACK_KEYSET, x)


sigmoid_native = _native_fallback


def sigmoid_triton(x: torch.Tensor) -> torch.Tensor:
    """计算逐元素 sigmoid；fp16/bf16/fp32 使用 fp32 中间值。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"sigmoid expects a Tensor, got {type(x)!r}")
    if x.layout != torch.strided or x.dtype not in _SUPPORTED:
        return _native_fallback(x)

    # P800 pointwise_dynamic 和 XMLIR 原生 sigmoid 的 0D launch 都会停在
    # 设备同步；按长度 1 向量执行后再恢复标量 shape。空张量不发射 kernel。
    if x.numel() == 0:
        return torch.empty_like(x)
    scalar_input = x.ndim == 0
    dispatch_input = x.reshape(1) if scalar_input else x
    def restore_shape(result: torch.Tensor) -> torch.Tensor:
        return result.reshape(()) if scalar_input else result

    # FlagGems5/XMLIR 的 pointwise_dynamic 和常规 masked Triton kernel
    # 在 255 元素边界均会停在设备同步。这里不做设备侧 padding/copy：
    # 直接以 256 lane 无 mask 访存计算，第 256 lane 的输入下标钳制到
    # 最后一个合法元素，再裁剪扩展输出。只在连续输入上启用；非连续输入
    # 沿用下方平台通用路径。
    if dispatch_input.numel() == 255 and dispatch_input.is_contiguous():
        out_storage = torch.empty(
            256, dtype=dispatch_input.dtype, device=dispatch_input.device)
        with torch_device_fn.device(dispatch_input.device):
            _sigmoid_255_no_mask_kernel[(1,)](dispatch_input, out_storage)
        return out_storage[:255].reshape(x.shape)

    try:
        from .backends import get_impl
    except ImportError:
        from kernel.backends import get_impl
    platform_impl = get_impl(dispatch_input.device.type)
    if platform_impl is not None:
        result = platform_impl(dispatch_input)
        return restore_shape(result)

    source = (dispatch_input if dispatch_input.is_contiguous()
              else dispatch_input.contiguous())
    out = torch.empty_like(source)
    n_elements = source.numel()

    # P800 的 Triton 后端适合较大的向量 program；与仓库已验证的
    # pointwise/clone 范式一致，大张量用 65536，避免数万个 program。
    block_size = (65536 if n_elements >= 65536
                  else max(256, triton.next_power_of_2(n_elements)))
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(source.device):
        _sigmoid_kernel[grid](source, out, n_elements, BLOCK_SIZE=block_size)
    return restore_shape(out)
