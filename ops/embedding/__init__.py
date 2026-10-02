"""ops.embedding — aten::embedding 算子包（包化同 ops.sdpa）。"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module(f"ops.embedding.{name}")


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "embedding":
        return _pkg_import("kernel.ascend910").embedding
    if name == "reference":
        return _pkg_import("reference").embedding_reference
    raise AttributeError(f"ops.embedding has no attribute {name!r}")


__all__ = ["register_a1", "embedding", "reference"]
