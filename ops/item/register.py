# 注册: 路线 A1（aten::item 可被 PrivateUse1 拦截）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"item": 0}
_BASE = os.environ.get("A1_ITEM_COUNT_FILE")
COUNT_FILE = f"{_BASE}.{os.getpid()}" if _BASE else None
_LIB = None


def _bump():
    CALL_COUNT["item"] = CALL_COUNT.get("item", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["item"] = counts.get("item", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def item_aten(self):
    _bump()
    from kernel.torch_level import item_torch
    return item_torch(self)


def register_a1(dispatch_key="PrivateUse1"):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("item", item_aten, dispatch_key)
    return _LIB


register = register_a1
