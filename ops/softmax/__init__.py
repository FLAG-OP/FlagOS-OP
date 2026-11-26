"""ops.softmax — aten::softmax 算子包（issue #55, 平头哥 thead-ppu）。"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.softmax." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").softmax_reference
    if name == "softmax_ppu":
        return _pkg_import("kernel.thead_ppu").softmax_ppu
    if name == "_softmax_ppu":
        return _pkg_import("kernel.thead_ppu")._softmax_ppu
    raise AttributeError(f"ops.softmax has no attribute {name!r}")


__all__ = ["register_a1", "reference", "softmax_ppu", "_softmax_ppu"]
