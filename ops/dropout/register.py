# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"dropout": 0}
_COUNT_BASE = os.environ.get("A1_DROPOUT_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["dropout"] = CALL_COUNT.get("dropout", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["dropout"] = counts.get("dropout", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def dropout_aten(input, p: float, train: bool):
    """aten::dropout(Tensor input, float p, bool train) -> Tensor"""
    _bump()
    from kernel.triton_level import dropout_triton
    return dropout_triton(input, p, train)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("dropout", dropout_aten, dispatch_key)
    return _LIB


register = register_a1
