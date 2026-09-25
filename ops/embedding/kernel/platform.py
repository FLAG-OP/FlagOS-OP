"""Multi-platform facade for ``aten::embedding``.

Mirrors ``ops/sdpa/kernel/triton_level.py``: callers keep a single import and
the platform backend is chosen at call time from ``tensor.device.type``.

- ``mlu``  -> [kernel/cambricon.py](cambricon.py) (Cambricon MLU590, torch_mlu)
- ``cuda`` -> [kernel/p800_kunlunxin.py](p800_kunlunxin.py) (XMLIR presents
  the Kunlunxin XPU as a CUDA device)

Tests, benchmark scripts, and the A1 registration in ``register.py`` import
from this module so a single code path serves both platforms.
"""
from __future__ import annotations

import torch

try:  # Package-style import: ops.embedding.kernel.platform
    from . import cambricon as _cambricon
    from . import p800_kunlunxin as _p800
except ImportError:  # Standalone import with OP_DIR on sys.path
    from kernel import cambricon as _cambricon
    from kernel import p800_kunlunxin as _p800


PLATFORM = "multi(cambricon,p800-kunlunxin)"
SUPPORTED_DEVICE_TYPES = ("mlu", "cuda")

_BACKENDS = {"mlu": _cambricon, "cuda": _p800}

# A1 拦截 key 与设备类型对应（实测结论见 reports/cambricon.md §4）:
# 两个平台都必须注册到各自的 Autograd key，autograd.Function 才能同时
# 覆盖 grad/no_grad 两种调用（p800=AutogradCUDA，MLU=AutogradPrivateUse1）。
DISPATCH_KEYS = {"cuda": "AutogradCUDA", "mlu": "AutogradPrivateUse1"}


def get_backend(device_type: str):
    """Return the platform backend module for a torch device type."""
    try:
        return _BACKENDS[device_type]
    except KeyError:
        raise RuntimeError(
            f"embedding 不支持 device_type={device_type!r}，"
            f"已知平台设备类型: {sorted(_BACKENDS)}"
        ) from None


def dispatch_key_for(device: str | torch.device) -> str:
    """设备字符串 -> A1 拦截 key（bench 脚本没有 profile 时用）。"""
    device_type = str(device).split(":")[0]
    if device_type not in DISPATCH_KEYS:
        raise RuntimeError(
            f"没有 {device_type!r} 对应的 A1 dispatch key，"
            f"已知: {DISPATCH_KEYS}"
        )
    return DISPATCH_KEYS[device_type]


def synchronize(device: str | torch.device) -> None:
    """平台同步（计时口径: kernel 完成后再读回/再取时间）。"""
    device_type = str(device).split(":")[0]
    if device_type == "mlu":
        torch.mlu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()
    else:
        raise RuntimeError(
            f"没有 {device_type!r} 对应的 synchronize 实现"
        )


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """``aten::embedding`` routed to the backend of ``weight.device``."""
    return get_backend(weight.device.type).embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )


def embedding_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Dense embedding backward routed by ``grad_output.device``."""
    return get_backend(grad_output.device.type).embedding_backward(
        grad_output, indices, num_weights, padding_idx,
        scale_grad_by_freq, sparse,
    )
