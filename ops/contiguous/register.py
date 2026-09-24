# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"contiguous": 0}
_COUNT_BASE = os.environ.get("A1_CONTIGUOUS_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["contiguous"] = CALL_COUNT.get("contiguous", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["contiguous"] = counts.get("contiguous", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def contiguous_aten(self, *, memory_format=torch.contiguous_format):
    """aten::contiguous(Tensor self, *, MemoryFormat memory_format=...)"""
    _bump()
    from kernel.triton_level import contiguous_triton
    return contiguous_triton(self, memory_format)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("contiguous", contiguous_aten, dispatch_key)
    return _LIB


register = register_a1
