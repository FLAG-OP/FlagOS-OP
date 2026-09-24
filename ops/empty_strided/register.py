# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
# empty_strided 可安全注册：实现经 empty + as_strided（不调用自身）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"empty_strided": 0}
_COUNT_BASE = os.environ.get("A1_EMPTY_STRIDED_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["empty_strided"] = CALL_COUNT.get("empty_strided", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["empty_strided"] = counts.get("empty_strided", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def empty_strided_aten(size, stride, *, dtype=None, layout=None, device=None,
                       pin_memory=None):
    """aten::empty_strided(SymInt[] size, SymInt[] stride, *, ...) -> Tensor"""
    _bump()
    from kernel.torch_level import empty_strided_torch
    return empty_strided_torch(size, stride, dtype=dtype, layout=layout,
                               device=device, pin_memory=pin_memory)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("empty_strided", empty_strided_aten, dispatch_key)
    return _LIB


register = register_a1
