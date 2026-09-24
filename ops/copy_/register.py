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
    from kernel.triton_level import copy_triton
    return copy_triton(self, src, non_blocking)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("copy_", copy_aten, dispatch_key)
    return _LIB


register = register_a1
