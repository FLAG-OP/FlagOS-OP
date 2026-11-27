"""ops.linear — aten::linear 算子包（issue #55, 平头哥 thead-ppu）。

包化约定（多算子同进程必用, 见 ops/README「包化导入」）:
  - 包态: from ops.linear import register_a1, reference
  - 脚本态: sys.path.insert(0, 算子目录) 后 from register import ...
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.linear." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").linear_reference
    if name == "linear_ppu":
        return _pkg_import("kernel.thead_ppu").linear_ppu
    if name == "linear_torch":
        return _pkg_import("kernel.torch_level").linear_torch
    raise AttributeError(f"ops.linear has no attribute {name!r}")


__all__ = ["register_a1", "reference", "linear_ppu", "linear_torch"]
