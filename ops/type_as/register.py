# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# type_as 是标准 aten 算子，宿主决定交付路径 = A1（本库集成指南）。
# dispatch key 由设备 profile 提供（MLU = PrivateUse1）。
#
# 调用计数（内存 + 可选文件落地）: 框架级测试跨进程读取，pid 分片避免
# TP 多 worker 并发写竞争。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"type_as": 0}
_COUNT_BASE = os.environ.get("A1_TYPE_AS_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["type_as"] = CALL_COUNT.get("type_as", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["type_as"] = counts.get("type_as", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def type_as_aten(self, other):
    """aten::type_as(Tensor self, Tensor other) -> Tensor"""
    _bump()
    try:
        from .kernel.triton_level import type_as_triton
    except ImportError:  # standalone single-operator script mode
        from kernel.triton_level import type_as_triton
    return type_as_triton(self, other)


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
    """Register ``aten::type_as`` with optional interception counter.

    ``platform`` is accepted explicitly so a second backend can later use the
    same call contract. These operators currently have one Cambricon backend.
    """
    _backend_for(dispatch_key, platform)

    def aten_impl(*args, **kwargs):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return type_as_aten(*args, **kwargs)

    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("type_as", aten_impl, dispatch_key)
    return _LIB


register = register_a1
