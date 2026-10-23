"""A1 aten::matmul operator package."""
from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name == "register_a1":
        return import_module(".register", __name__).register_a1
    raise AttributeError(f"{__name__} has no attribute {name!r}")


__all__ = ["register_a1"]
