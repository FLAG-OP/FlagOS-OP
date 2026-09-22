"""SDPA platform backends.

The public entry is still ``kernel.triton_level.sdpa_triton``.  This package
keeps per-device implementations isolated while preserving the original import
path used by tests and user code.
"""
from __future__ import annotations

import importlib


_IMPL_BY_DEVICE = {
    "npu": "ascend910",
    # Kunlunxin XMLIR presents its XPU devices as CUDA tensors.  The backend
    # itself performs a torch_xmlir guard, so an ordinary CUDA build is rejected.
    "cuda": "p800_kunlunxin",
}


def get_impl(device_type: str):
    module_name = _IMPL_BY_DEVICE.get(device_type)
    if module_name is None:
        raise RuntimeError(
            f"sdpa 无 {device_type!r} 平台实现，已注册: "
            f"{list(_IMPL_BY_DEVICE)}（见 MERGE.md / PLATFORM.md）")
    # Lazy import avoids loading triton-ascend on Kunlunxin and vice versa.
    return importlib.import_module(f".{module_name}", __package__)


__all__ = ["get_impl"]
