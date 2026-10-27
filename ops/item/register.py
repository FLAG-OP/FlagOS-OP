# 注册: 路线 A1（aten::item 可被 PrivateUse1 拦截）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"item": 0}
_BASE = os.environ.get("A1_ITEM_COUNT_FILE")
COUNT_FILE = f"{_BASE}.{os.getpid()}" if _BASE else None
_LIB = None


def _bump():
    CALL_COUNT["item"] = CALL_COUNT.get("item", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["item"] = counts.get("item", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def item_aten(self):
    _bump()
    # 包态优先、脚本态 fallback（多算子同进程执行，#25 review）
    try:
        from .kernel.torch_level import item_torch
    except ImportError:
        from kernel.torch_level import item_torch
    return item_torch(self)


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
    # 平台由 profile 显式传入（profile.vendor）；不再按设备探测自动选择。
    # Dashener2 2026-10-20-fix: 按 review #34 收敛（避免“探测到哪个设备就优先选哪个”）。
    if platform is None:
        raise RuntimeError(
            "register_a1 需要显式 platform（应为 profile.vendor，如 "
            f"{sorted(_BACKENDS)}）；dispatch_key={dispatch_key!r} "
            "已不再自动推断（见 ops/embedding/PLATFORM.md §5）")
    if platform not in _BACKENDS:
        raise RuntimeError(
            f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
            "（新平台接入见 ops/embedding/PLATFORM.md §5）")
    return _BACKENDS[platform]


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
    """注册 item 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradPrivateUse1（对齐模板）；MLU 实际拦截 key
        为 profile 的 PrivateUse1，两 key 均可命中，platform 显式优先。
    counter: 可选 dict，命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    _probe_guard(backend, dispatch_key)
    fn = item_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return item_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    for _k in _a1_keys(dispatch_key):
        _LIB.impl("item", fn, _k)
    return _LIB


register = register_a1
