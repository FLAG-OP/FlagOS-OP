"""ops._softmax — aten::_softmax 私有原语包（issue #55）。

与 ops.softmax 的关系（issue #55 要求记录）: F.softmax → aten::softmax
（ops/softmax）→ 内部落点 aten::_softmax（本目录, 委托原生）。
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops._softmax." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").softmax_prim_reference
    if name == "_softmax_ppu":
        return _pkg_import("kernel.thead_ppu")._softmax_ppu
    raise AttributeError(f"ops._softmax has no attribute {name!r}")


__all__ = ["register_a1", "reference", "_softmax_ppu"]
