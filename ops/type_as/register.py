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
    from kernel.triton_level import type_as_triton
    return type_as_triton(self, other)


def register_a1(dispatch_key: str = "PrivateUse1") -> torch.library.Library:
    """注册 type_as 到 aten；Library 对象必须保持引用（否则被回收）。"""
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("type_as", type_as_aten, dispatch_key)
    return _LIB


# 统一入口别名（与 A2/B 插件约定对齐，便于框架级注入）
register = register_a1
