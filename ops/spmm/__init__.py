"""ops.spmm — CSR 稀疏×稠密矩阵乘（aten::_sparse_mm; issue #57）。

宿主绑定: torch.sparse.mm → aten::_sparse_mm; torch.spmm = torch.mm
别名（保留稠密行为, 不注册）。Gowalla 训练链路的交付说明见 README.md。
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.spmm." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").spmm_reference
    if name == "spmm_ppu":
        return _pkg_import("kernel.thead_ppu").spmm_ppu
    raise AttributeError(f"ops.spmm has no attribute {name!r}")


__all__ = ["register_a1", "reference", "spmm_ppu"]
