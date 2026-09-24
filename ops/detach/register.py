# 注册: 路线 A1（aten::detach 可被 PrivateUse1 拦截）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"detach": 0}
_BASE = os.environ.get("A1_DETACH_COUNT_FILE")
COUNT_FILE = f"{_BASE}.{os.getpid()}" if _BASE else None
_LIB = None


def _bump():
    CALL_COUNT["detach"] = CALL_COUNT.get("detach", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["detach"] = counts.get("detach", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def detach_aten(self):
    _bump()
    from kernel.torch_level import detach_torch
    return detach_torch(self)


def register_a1(dispatch_key="PrivateUse1"):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("detach", detach_aten, dispatch_key)
    return _LIB


register = register_a1
