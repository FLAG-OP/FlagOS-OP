"""SDPA platform backends.

The public entry is still ``kernel.triton_level.sdpa_triton``.  This package
keeps per-device implementations isolated while preserving the original import
path used by tests and user code.
"""
from __future__ import annotations

import importlib
import importlib.util


# device.type -> backend module.  "cuda" 有三个候选（昆仑芯 XMLIR 伪装
# CUDA / T-head PPU 的 CUDA 兼容层 / NVIDIA），见 _cuda_impl_name。
_IMPL_BY_DEVICE = {
    "npu": "ascend910",
    # Kunlunxin XMLIR presents its XPU devices as CUDA tensors.  The backend
    # itself performs a torch_xmlir guard, so an ordinary CUDA build is rejected.
    "cuda": "p800_kunlunxin",
    # Cambricon torch_mlu tensors use device.type == "mlu".
    "mlu": "mlu590",
}


def _stack_is_ppu() -> bool:
    """PPU 栈识别（与 device type 无关, 仅用于 cuda 三候选的路由）。

    证据: ppu_compile_evidence.json 记录 triton driver_type=PPUDriver、
    设备名 PPU-ZW810E；两路任一命中即认为当前是 PPU 栈。
    """
    try:
        from triton.runtime import driver
        if "PPU" in type(driver.active).__name__:
            return True
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available() and any(
            "PPU" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


def _cuda_impl_name() -> str | None:
    # XMLIR 先判（p800 backend 内部同样有 torch_xmlir 守卫）。
    if importlib.util.find_spec("torch_xmlir") is not None:
        try:
            import torch_xmlir  # noqa: F401
            return "p800_kunlunxin"
        except Exception:
            pass
    if _stack_is_ppu():
        return "thead_ppu"
    return None


# 每 device_type 一份解析缓存（栈身份进程内不变）。不缓存时
# find_spec/importlib 每调用 ~80-100µs CPU 开销——thead PPU 实测把
# flash 直调 0.023ms 抬到 0.061ms（cProfile 定位 _cuda_impl_name），
# 是"委托比原生慢 3×"倒挂的主因之一。
_IMPL_CACHE: dict = {}


def get_impl(device_type: str):
    cached = _IMPL_CACHE.get(device_type)
    if cached is not None:
        return cached
    if device_type == "cuda":
        module_name = _cuda_impl_name()
    else:
        module_name = _IMPL_BY_DEVICE.get(device_type)
    if module_name is None:
        raise RuntimeError(
            f"sdpa 无 {device_type!r} 平台实现，已注册: "
            f"{list(_IMPL_BY_DEVICE)}（见 MERGE.md / PLATFORM.md；"
            f"cuda 需为 torch_xmlir/P800 或 PPU 栈）")
    # Lazy import avoids loading triton-ascend on Kunlunxin and vice versa.
    mod = importlib.import_module(f".{module_name}", __package__)
    _IMPL_CACHE[device_type] = mod
    return mod


__all__ = ["get_impl"]
