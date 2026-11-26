"""ops.matmul — aten::matmul 算子包（issue #55; 公共骨架, PPU 首交付）。

⚠️ 公共目录: PR #35（P800）/ PR #53（MLU590）在途——合入时按 MERGE.md
七步流程合并平台范围, 不互相推广（issue #55 明确排除项条款）。
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.matmul." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").matmul_reference
    if name == "matmul_ppu":
        return _pkg_import("kernel.thead_ppu").matmul_ppu
    raise AttributeError(f"ops.matmul has no attribute {name!r}")


__all__ = ["register_a1", "reference", "matmul_ppu"]
