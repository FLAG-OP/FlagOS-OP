# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# empty_like 可安全注册：实现经 empty_strided 分配（不调用自身，无递归）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"empty_like": 0}
_COUNT_BASE = os.environ.get("A1_EMPTY_LIKE_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["empty_like"] = CALL_COUNT.get("empty_like", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["empty_like"] = counts.get("empty_like", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def empty_like_aten(self, *, dtype=None, layout=None, device=None,
                    pin_memory=None, memory_format=None):
    """aten::empty_like(Tensor self, *, ...) -> Tensor"""
    _bump()
    # 包态优先、脚本态 fallback（多算子同进程执行，#25 review）
    try:
        from .kernel.torch_level import empty_like_torch
    except ImportError:
        from kernel.torch_level import empty_like_torch
    return empty_like_torch(self, dtype=dtype, layout=layout, device=device,
                            pin_memory=pin_memory,
                            memory_format=memory_format)


# ── 平台选择（多平台: cambricon/MLU590 + ascend910）──
# Dashener2 2026-10-09-fix register_a1 签名升级（issue #19）: counter + platform，
# 默认 dispatch_key 对齐模板（AutogradPrivateUse1）。
# Dashener2 2026-10-15-fix 接入第二平台 ascend910（见 PLATFORM.md）: torch_npu 与
# torch_mlu 的 A1 拦截 key 同为 AutogradPrivateUse1（C++ Autograd 包装不
# redispatch，见 ops/sdpa/PLATFORM.md §2），故该 key 按「可用设备」消歧；
# platform 显式参数始终优先。
# # Dashener2 <dashen272@outlook.com>
_BACKENDS = {"cambricon": "cambricon", "ascend": "ascend"}
_PROBE_DEVICE = {"cambricon": "mlu", "ascend": "npu"}
_PROBE_MODULE = {"cambricon": "torch_mlu", "ascend": "torch_npu"}


def _probe_ok(backend: str) -> bool:
    import importlib
    try:
        importlib.import_module(_PROBE_MODULE[backend])  # 触发 torch.<dev> 注册
    except Exception:
        return False
    dev = getattr(torch, _PROBE_DEVICE[backend], None)
    return dev is not None and dev.is_available()


def _backend_for(dispatch_key: str, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（新平台接入见 ops/embedding/PLATFORM.md §5）")
        return _BACKENDS[platform]
    by_key = {
        "AutogradPrivateUse1": None,   # npu 与 mlu 同 key: 按可用设备消歧
        "PrivateUse1": None,           # torch_mlu 后端 key; 同样按设备消歧
    }
    if dispatch_key not in by_key:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: {list(by_key)}"
            "（或显式传 platform）")
    backend = by_key[dispatch_key]
    if backend is None:
        if _probe_ok("ascend"):
            backend = "ascend"
        elif _probe_ok("cambricon"):
            backend = "cambricon"
        else:
            raise RuntimeError(
                f"dispatch_key={dispatch_key!r} 需要可用的 Ascend(torch_npu) "
                "或 MLU(torch_mlu) 设备（或显式传 platform）")
    return backend


def _probe_guard(backend: str, dispatch_key: str) -> None:
    if not _probe_ok(backend):
        dev_name = {"cambricon": "MLU（torch.mlu.is_available()）",
                    "ascend": "Ascend（torch.npu.is_available()）"}
        raise RuntimeError(
            f"register_a1 backend={backend!r}, dispatch_key={dispatch_key!r} "
            f"需要可用的 {dev_name.get(backend, backend)} 设备")


# A1 注册键成对覆盖（sdpa_math 范式）: torch_npu 上不同算子命中的 key 不同——
# 带 autograd 的算子落 AutogradPrivateUse1，工厂算子（empty_strided）落
# PrivateUse1；torch.inference_mode() 下 Autograd* 也可能不命中。故成对注册。
_A1_KEY_PAIRS = {"AutogradPrivateUse1": "PrivateUse1",
                 "PrivateUse1": "AutogradPrivateUse1"}


def _a1_keys(dispatch_key: str) -> list:
    keys = [dispatch_key]
    if dispatch_key in _A1_KEY_PAIRS:
        keys.append(_A1_KEY_PAIRS[dispatch_key])
    return keys


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 empty_like 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradPrivateUse1（对齐模板）；MLU 实际拦截 key
        为 profile 的 PrivateUse1，两 key 均可命中，platform 显式优先。
    counter: 可选 dict，命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    _probe_guard(backend, dispatch_key)
    fn = empty_like_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return empty_like_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    for _k in _a1_keys(dispatch_key):
        _LIB.impl("empty_like", fn, _k)
    return _LIB


register = register_a1
