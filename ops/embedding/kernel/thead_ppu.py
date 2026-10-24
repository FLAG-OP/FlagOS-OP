"""T-head PPU (ZW810E) embedding backend.

首测结论（probes/thead_first_probe.py, 2026-10-23, 卡6）:

- ``aten::embedding`` / ``index_select`` / ``embedding_backward`` overload
  均存在（schema 探测 + 直调实测）。
- native ``embedding`` vs CPU 官方: fp32/fp16/bf16 × padding ∈ {-1, 3}
  全部 **err=0.0**（逐位一致）；``index_select`` 委托面同样 err=0.0。
- ``embedding_backward``: ``scale_grad_by_freq=True`` 原生完整
  （fp16 对 CPU fp32 参考 err 6.1e-5 ~ 1.2e-4，属 fp16 累加序噪声）——
  与 cambricon 同为"原生完整"型，**无需** p800/XPU 的 inverse-frequency
  补偿路径。
- dispatch key: ``AutogradCUDA`` 6/7 命中（唯一零命中 inference_mode =
  known-issues #19 的 Autograd 键通病, 与 T1 sdpa 实测同）；纯 ``CUDA``
  键 7/7 全命中但按 p800/T1 先例取 Autograd key 以携带自带 backward。

生产路径与 p800/cambricon 同构: forward 委托 ``aten::index_select``
（绕开 A1 注册后的 dispatcher 递归），dense backward 委托
``aten::embedding_backward``（原生完整, scale_grad_by_freq 直通）。
Triton gather 不接生产——P800/MLU 两平台实测慢 11-102x（README Triton
实验节），embedding 是带宽型算子，自研 Triton 无收益。
"""
from __future__ import annotations

import torch


PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # PPU torch 经 CUDA 兼容层直曝 cuda:N
_HAS_PPU_STACK = None


def _has_ppu_stack() -> bool:
    """进程级 PPU 栈识别（与 T1 sdpa _stack_is_ppu 同款双路探测）。"""
    global _HAS_PPU_STACK
    if _HAS_PPU_STACK is None:
        try:
            from triton.runtime import driver
            if "PPU" in type(driver.active).__name__:
                _HAS_PPU_STACK = True
                return True
        except Exception:
            pass
        try:
            _HAS_PPU_STACK = torch.cuda.is_available() and any(
                "PPU" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            )
        except Exception:
            _HAS_PPU_STACK = False
    return _HAS_PPU_STACK


def _device_is_ppu(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return "PPU" in torch.cuda.get_device_name(device)
    except Exception:
        return False


def _validate_device_dtype(primary, indices, padding_idx, num_weights):
    if (primary.device.type not in SUPPORTED_DEVICE_TYPES
            or not _device_is_ppu(primary.device)):
        raise RuntimeError(
            f"embedding 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={primary.device!r}（非 PPU 栈）。"
        )
    if primary.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported floating dtype: {primary.dtype}")
    if indices.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"unsupported indices dtype: {indices.dtype}")
    if primary.device != indices.device:
        raise ValueError("weight and indices must be on the same device")
    if padding_idx is not None and padding_idx != -1:
        if padding_idx < 0 or padding_idx >= num_weights:
            raise IndexError(
                "padding_idx must be -1/None or inside [0, num_weights)"
            )


def _validate_forward(weight, indices, padding_idx, sparse):
    # ``sparse`` only requests a sparse weight gradient.  It does not change
    # the forward lookup, so sparse=True is accepted here and rejected only by
    # embedding_backward.
    _validate_device_dtype(weight, indices, padding_idx, weight.shape[0])
    if weight.dim() != 2:
        raise ValueError("weight must be 2D (num_weights, dim)")


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Native row-gather implementation of ``aten::embedding``."""
    # Hot path for the common 1D token-id call: one native row gather, no
    # Python reshape/view, and no semantic transform from embedding options.
    if (
        _device_is_ppu(weight.device)
        and weight.dim() == 2
        and indices.dim() == 1
        and padding_idx in (-1, None)
    ):
        return torch.ops.aten.index_select(weight, 0, indices)
    _validate_forward(weight, indices, padding_idx, sparse)
    # index_select requires a vector while aten::embedding accepts any index
    # shape.  Flatten here and restore the exact output shape.
    flat_indices = (
        indices if indices.dim() == 1 else indices.reshape(-1)
    )
    out = torch.ops.aten.index_select(weight, 0, flat_indices)
    if flat_indices is indices:
        return out
    return out.view(*indices.shape, weight.shape[-1])


def embedding_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Dense backward delegating to the complete PPU native kernel."""
    _validate_device_dtype(
        grad_output, indices, padding_idx, num_weights
    )
    if num_weights < 0:
        raise ValueError("num_weights must be non-negative")
    if sparse:
        # 本交付只覆盖 dense backward; sparse 梯度格式是显式不支持边界
        # （kernel 层 guard 测试依赖此行为, 与 cambricon/p800 同口径）。
        raise NotImplementedError(
            "sparse embedding backward is outside this delivery"
        )
    if (grad_output.dim() != indices.dim() + 1
            or grad_output.shape[:-1] != indices.shape):
        raise ValueError("grad_output shape must be (*indices, dim)")

    # PPU 原生 embedding_backward 完整实现 padding_idx 与
    # scale_grad_by_freq（首测 err 6.1e-5 ~ 1.2e-4 fp16 累加序噪声,
    # probes/thead_first_overloads.json）——与 cambricon 同为直接委托,
    # 无 p800/XPU 的 inverse-frequency 补偿。
    return torch.ops.aten.embedding_backward(
        grad_output, indices, num_weights, padding_idx,
        scale_grad_by_freq=scale_grad_by_freq, sparse=False,
    )


# 注册守卫用能力探测声明 (attr, fn)——须同时验证 cuda 可用与 PPU 栈身份
# （仅 torch.cuda.is_available() 在 NVIDIA 机器上也为真, 不足以守卫）。
_DEVICE_PROBE = (
    "cuda",
    (lambda: torch.cuda.is_available() and any(
        "PPU" in torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    )),
)
