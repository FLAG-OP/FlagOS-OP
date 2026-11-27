"""``ops.sigmoid`` — ``aten::sigmoid`` A1 算子包。"""
from __future__ import annotations

import importlib

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>

PLATFORM = "multi-triton"


def _pkg_import(name: str):
    return importlib.import_module("ops.sigmoid." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").sigmoid_reference
    if name == "sigmoid_triton":
        return _pkg_import("kernel.triton_level").sigmoid_triton
    raise AttributeError(f"ops.sigmoid has no attribute {name!r}")


__all__ = ["PLATFORM", "register_a1", "reference", "sigmoid_triton"]
