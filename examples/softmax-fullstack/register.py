# A1 路线注册: aten::_softmax 拦截（从原 example.py 抽取）。
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel.triton_level import softmax_triton  # noqa: E402


# ============ aten 注册 ============
_LIB = None
CALLS = {"softmax": 0}

def _softmax_counter(self, dim=-1, dtype=None):
    CALLS["softmax"] += 1
    return softmax_triton(self, dim)

def register_softmax(dispatch_key: str):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("_softmax", _softmax_counter, dispatch_key)
