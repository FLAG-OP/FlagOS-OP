# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# ⚠️ copy_ 是极高频内部算子，注册后进程内所有 copy_ 都走本实现。测试须
# 在注册前取原生参考；示例编排里框架层用子进程隔离。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"copy_": 0}
_COUNT_BASE = os.environ.get("A1_COPY_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["copy_"] = CALL_COUNT.get("copy_", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["copy_"] = counts.get("copy_", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def copy_aten(self, src, non_blocking: bool = False):
    """aten::copy_(Tensor self, Tensor src, bool non_blocking=False) -> Tensor"""
    _bump()
    try:
        from .kernel.triton_level import copy_triton
    except ImportError:  # standalone single-operator script mode
        from kernel.triton_level import copy_triton
    return copy_triton(self, src, non_blocking)


_PLATFORM_ALIASES = {"mlu": "cambricon", "mlu590": "cambricon"}
_SUPPORTED_PLATFORMS = {"cambricon"}
_SUPPORTED_DISPATCH_KEYS = {
    "AutogradPrivateUse1", "PrivateUse1", "Autograd",
}


def _backend_for(dispatch_key: str, platform: str | None = None) -> str:
    """Resolve the sole current backend without silently accepting typos."""
    requested = platform or "cambricon"
    canonical = _PLATFORM_ALIASES.get(requested, requested)
    if canonical not in _SUPPORTED_PLATFORMS:
        raise RuntimeError(
            f"未知 platform={platform!r}; 已支持: "
            f"{sorted(_SUPPORTED_PLATFORMS)}")
    if dispatch_key not in _SUPPORTED_DISPATCH_KEYS:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: "
            f"{sorted(_SUPPORTED_DISPATCH_KEYS)}（或显式传 platform）")
    return canonical


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                platform: str | None = None) -> torch.library.Library:
    """Register ``aten::copy_`` with optional interception counter.

    ``platform`` is accepted explicitly so a second backend can later use the
    same call contract. These operators currently have one Cambricon backend.
    """
    _backend_for(dispatch_key, platform)

    def aten_impl(*args, **kwargs):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return copy_aten(*args, **kwargs)

    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("copy_", aten_impl, dispatch_key)
    return _LIB


register = register_a1
