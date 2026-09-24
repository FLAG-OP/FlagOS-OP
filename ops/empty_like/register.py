# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# empty_like 可安全注册：实现经 empty_strided 分配（不调用自身，无递归）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"empty_like": 0}
_COUNT_BASE = os.environ.get("A1_EMPTY_LIKE_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["empty_like"] = CALL_COUNT.get("empty_like", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["empty_like"] = counts.get("empty_like", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def empty_like_aten(self, *, dtype=None, layout=None, device=None,
                    pin_memory=None, memory_format=None):
    """aten::empty_like(Tensor self, *, ...) -> Tensor"""
    _bump()
    from kernel.torch_level import empty_like_torch
    return empty_like_torch(self, dtype=dtype, layout=layout, device=device,
                            pin_memory=pin_memory,
                            memory_format=memory_format)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("empty_like", empty_like_aten, dispatch_key)
    return _LIB


register = register_a1
