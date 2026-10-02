"""ops.sdpa_math — private SDPA math operator package."""
from __future__ import annotations


def _pkg_import(name: str):
    import importlib
    return importlib.import_module(f"ops.sdpa_math.{name}")


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "sdpa_math_triton":
        return _pkg_import("kernel.triton_level").sdpa_math_triton
    if name == "sdpa_math_p800":
        return _pkg_import("kernel.p800_fast_level").sdpa_math_p800_fast
    if name == "sdpa_math_torch":
        return _pkg_import("kernel.torch_level").sdpa_math_torch
    if name == "reference":
        return _pkg_import("reference").sdpa_math_reference
    if name == "PLATFORM":
        return _pkg_import("kernel.triton_level").PLATFORM
    raise AttributeError(f"ops.sdpa_math has no attribute {name!r}")


__all__ = [
    "register_a1", "sdpa_math_triton", "sdpa_math_p800",
    "sdpa_math_torch", "reference", "PLATFORM",
]
