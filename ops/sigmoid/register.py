"""路线 A1：把 sigmoid 路由注册到 ``aten::sigmoid``。"""
from __future__ import annotations

import json
import os

import torch

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>
# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>

CALL_COUNT = {"sigmoid": 0}
_COUNT_BASE = os.environ.get("A1_SIGMOID_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None
_LIBS: dict[str, torch.library.Library] = {}
_REGISTERED_KEYS: tuple[str, ...] = ()
_ACTIVE_IMPL = "native"
_COUNTERS = []
_SUPPORTED_PLATFORMS = {
    "kunlunxin", "p800-kunlunxin", "p800-kunlunxin-gems5",
}


def _load_impl(name: str):
    try:
        from .kernel.triton_level import sigmoid_native, sigmoid_triton
    except ImportError:
        from kernel.triton_level import sigmoid_native, sigmoid_triton
    implementations = {
        "native": sigmoid_native,
        "triton": sigmoid_triton,
    }
    try:
        return implementations[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown sigmoid impl={name!r}; supported: "
            f"{sorted(implementations)}"
        ) from exc


def _bump() -> None:
    CALL_COUNT["sigmoid"] = CALL_COUNT.get("sigmoid", 0) + 1
    for counter in _COUNTERS:
        counter["n"] = counter.get("n", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE, encoding="utf-8") as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["sigmoid"] = counts.get("sigmoid", 0) + 1
        with open(COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f)


def sigmoid_aten(self: torch.Tensor) -> torch.Tensor:
    _bump()
    return _load_impl(_ACTIVE_IMPL)(self)


def register_a1(
    dispatch_key: str = "AutogradCUDA", counter: dict | None = None,
    impl: str = "native", platform: str | None = None,
) -> list[torch.library.Library]:
    """注册一次并保持 Library 生命周期。

    ``AutogradCUDA`` 是 P800 A1 canonical key，``CUDA`` 是兼容 key；
    两者始终成对注册，覆盖 grad/no_grad 与 inference_mode。默认 ``native``
    路由保持生产语义和性能；开发验证显式传 ``impl="triton"``。

    ``counter`` 遵循仓库 A1 契约，在命中时增加 ``counter["n"]``；
    ``platform`` 显式限定当前已交付的昆仑芯 P800 平台。
    """
    global _ACTIVE_IMPL, _REGISTERED_KEYS
    pairs = {
        "AutogradCUDA": ("AutogradCUDA", "CUDA"),
        "CUDA": ("AutogradCUDA", "CUDA"),
    }
    if dispatch_key not in pairs:
        raise RuntimeError(
            "sigmoid P800 supports dispatch_key='AutogradCUDA' (canonical) "
            f"or 'CUDA' (compatibility), got {dispatch_key!r}"
        )
    if platform is not None and platform not in _SUPPORTED_PLATFORMS:
        raise RuntimeError(
            f"unknown platform={platform!r}; supported: "
            f"{sorted(_SUPPORTED_PLATFORMS)}"
        )
    if counter is not None and counter not in _COUNTERS:
        _COUNTERS.append(counter)
    _load_impl(impl)  # validate before mutating registration state
    keys = pairs[dispatch_key]
    if _LIBS:
        if keys != _REGISTERED_KEYS or impl != _ACTIVE_IMPL:
            raise RuntimeError(
                f"sigmoid already registered for {_REGISTERED_KEYS} with "
                f"impl={_ACTIVE_IMPL!r}; cannot switch to {keys}/impl={impl!r}"
            )
        return [_LIBS[key] for key in _REGISTERED_KEYS]

    _ACTIVE_IMPL = impl
    _REGISTERED_KEYS = keys
    for key in keys:
        lib = torch.library.Library("aten", "IMPL")
        lib.impl("sigmoid", sigmoid_aten, key)
        _LIBS[key] = lib
    return [_LIBS[key] for key in _REGISTERED_KEYS]


register = register_a1
