# 注册: 路线 A1（aten::_local_scalar_dense 可被 PrivateUse1 拦截）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"_local_scalar_dense": 0}
_BASE = os.environ.get("A1_LSD_COUNT_FILE")
COUNT_FILE = f"{_BASE}.{os.getpid()}" if _BASE else None
_LIB = None


def _bump():
    k = "_local_scalar_dense"
    CALL_COUNT[k] = CALL_COUNT.get(k, 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts[k] = counts.get(k, 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def lsd_aten(self):
    _bump()
    from kernel.torch_level import _local_scalar_dense_torch
    return _local_scalar_dense_torch(self)


def register_a1(dispatch_key="PrivateUse1"):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("_local_scalar_dense", lsd_aten, dispatch_key)
    return _LIB


register = register_a1
