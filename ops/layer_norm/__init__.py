"""ops.layer_norm — aten::layer_norm 算子包（issue #55）。"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.layer_norm." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").layer_norm_reference
    if name == "layer_norm_ppu":
        return _pkg_import("kernel.thead_ppu").layer_norm_ppu
    raise AttributeError(f"ops.layer_norm has no attribute {name!r}")


__all__ = ["register_a1", "reference", "layer_norm_ppu"]
