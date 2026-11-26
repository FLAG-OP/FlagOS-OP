"""ops.native_layer_norm — aten::native_layer_norm 算子包（issue #55）。"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.native_layer_norm." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").native_layer_norm_reference
    if name == "native_layer_norm_ppu":
        return _pkg_import("kernel.thead_ppu").native_layer_norm_ppu
    raise AttributeError(
        f"ops.native_layer_norm has no attribute {name!r}")


__all__ = ["register_a1", "reference", "native_layer_norm_ppu"]
