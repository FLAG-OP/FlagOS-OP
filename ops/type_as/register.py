# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# type_as 是标准 aten 算子，宿主决定交付路径 = A1（本库集成指南）。
# dispatch key 由设备 profile 提供（MLU = PrivateUse1）。
#
# 调用计数（内存 + 可选文件落地）: 框架级测试跨进程读取，pid 分片避免
# TP 多 worker 并发写竞争。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"type_as": 0}
_COUNT_BASE = os.environ.get("A1_TYPE_AS_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["type_as"] = CALL_COUNT.get("type_as", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["type_as"] = counts.get("type_as", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def type_as_aten(self, other):
    """aten::type_as(Tensor self, Tensor other) -> Tensor"""
    _bump()
    from kernel.triton_level import type_as_triton
    return type_as_triton(self, other)


# ── 平台选择（单平台 cambricon/MLU590）──
# 第二平台接入时按 ops/embedding 范式扩成多后端 + PLATFORM.md/MERGE.md。
# Dashener2 2026-10-09-fix register_a1 签名升级（issue #19）: counter + platform，
# 默认 dispatch_key 对齐模板（AutogradPrivateUse1）；_DEVICE_PROBE 注册守卫。
# # Dashener2 <dashen272@outlook.com>
_DEVICE_PROBE = ("mlu", lambda: torch.mlu.is_available())
_BACKENDS = {"cambricon": "cambricon"}


def _backend_for(dispatch_key: str, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（第二平台接入见 ops/embedding/PLATFORM.md §5）")
        return _BACKENDS[platform]
    by_key = {
        "Autograd": "cambricon",
        "PrivateUse1": "cambricon",
        "AutogradPrivateUse1": "cambricon",
    }
    if dispatch_key not in by_key:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: {list(by_key)}"
            "（或显式传 platform）")
    return by_key[dispatch_key]


def _probe_ok() -> bool:
    attr, fn = _DEVICE_PROBE
    try:
        import torch_mlu  # noqa: F401  触发 torch.mlu 注册
    except Exception:
        return False
    return hasattr(torch, attr) and getattr(torch, attr) is not None and fn()


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 type_as 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradPrivateUse1（对齐模板）；MLU 实际拦截 key
        为 profile 的 PrivateUse1，两 key 均可命中，platform 显式优先。
    counter: 可选 dict，命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if backend == "cambricon" and not _probe_ok():
        raise RuntimeError(
            f"register_a1 backend={backend!r}, dispatch_key={dispatch_key!r} "
            "需要可用的 MLU 设备（torch.mlu.is_available()）")
    fn = type_as_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return type_as_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("type_as", fn, dispatch_key)
    return _LIB


# 统一入口别名（与 A2/B 插件约定对齐，便于框架级注入）
register = register_a1
