# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# ⚠️ copy_ 是极高频内部算子，注册后进程内所有 copy_ 都走本实现。测试须
# 在注册前取原生参考；示例编排里框架层用子进程隔离。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"copy_": 0}
_COUNT_BASE = os.environ.get("A1_COPY_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["copy_"] = CALL_COUNT.get("copy_", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["copy_"] = counts.get("copy_", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def copy_aten(self, src, non_blocking: bool = False):
    """aten::copy_(Tensor self, Tensor src, bool non_blocking=False) -> Tensor"""
    _bump()
    # 包态优先、脚本态 fallback（多算子同进程执行，#25 review）
    try:
        from .kernel.triton_level import copy_triton
    except ImportError:
        from kernel.triton_level import copy_triton
    return copy_triton(self, src, non_blocking)


# ── 平台选择（cambricon/MLU590 + p800-kunlunxin）──
# 按 ops/embedding 范式做多后端 + PLATFORM.md/MERGE.md。
# Dashener2 2026-10-09-fix register_a1 签名升级（issue #19）: counter + platform，
# 默认 dispatch_key 对齐模板（AutogradPrivateUse1）；_DEVICE_PROBE 注册守卫。
# # Dashener2 <dashen272@outlook.com>
_BACKENDS = {"cambricon": "cambricon", "p800-kunlunxin": "p800-kunlunxin"}


def _backend_for(dispatch_key: str, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（新平台接入见 ops/embedding/PLATFORM.md §5）")
        return _BACKENDS[platform]
    by_key = {
        "AutogradPrivateUse1": "cambricon",   # MLU canonical A1 key
        "PrivateUse1": "cambricon",           # MLU 兼容 key
        "AutogradCUDA": "p800-kunlunxin",     # P800 canonical A1 key
        "CUDA": "p800-kunlunxin",             # P800 兼容 key
    }
    if dispatch_key not in by_key:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: {list(by_key)}"
            "（或显式传 platform）")
    return by_key[dispatch_key]


def _probe_ok(backend: str) -> bool:
    if backend == "cambricon":
        try:
            import torch_mlu  # noqa: F401  触发 torch.mlu 注册
        except Exception:
            return False
        return hasattr(torch, "mlu") and torch.mlu.is_available()
    if backend == "p800-kunlunxin":
        try:
            import torch_xmlir  # noqa: F401  XPU 经 XMLIR 暴露为 cuda
        except Exception:
            return False
        return torch.cuda.is_available()
    return False


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 copy_ 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradPrivateUse1（对齐模板）；MLU 实际拦截 key
        为 profile 的 PrivateUse1，两 key 均可命中，platform 显式优先。
    counter: 可选 dict，命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r}, dispatch_key={dispatch_key!r} "
            f"需要可用的 {backend} 设备")
    fn = copy_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return copy_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("copy_", fn, dispatch_key)
    return _LIB


register = register_a1
