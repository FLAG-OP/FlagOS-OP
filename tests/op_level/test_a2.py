#!/usr/bin/env python3
"""A2 op 层测试: dispatch 注册 + 策略选择（kernel 直测见 kernel 层）。"""
from __future__ import annotations


def run(profile) -> bool:
    import os
    import time

    import torch
    from routes.a2_dispatch.plugin import kernels as K
    from vllm_fl.dispatch import get_default_manager, call_op
    from vllm_fl.dispatch.policy import with_preference

    dev = profile.torch_device
    # dispatch 插件自加载（直接调用本文件时也生效）。
    # 显式赋值 + 重置 manager: --all 模式下同进程串跑多个测试，
    # 避免上一个测试的插件残留/OpManager 缓存污染。
    os.environ["VLLM_FL_PLUGIN_MODULES"] = \
        "routes.a2_dispatch.plugin.register_ops"
    from vllm_fl.dispatch import reset_default_manager
    reset_default_manager()
    print("=" * 60)
    print(f"A2 op-level [{profile.name}]: gelu_and_mul 三段式 @ {dev}")
    print("=" * 60)

    # ---- dispatch 注册 + 策略切换 ----
    m = get_default_manager()
    m.ensure_initialized()
    impls = m.registry.snapshot().impls_by_op.get("gelu_and_mul", [])
    assert {i.impl_id for i in impls} >= {"default.flagos", "reference.torch"}
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev) * 2
    y = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    with with_preference("flagos"):
        call_op("gelu_and_mul", None, x, y)
        assert m._called_ops["gelu_and_mul"] == "default.flagos"
    with with_preference("reference"):
        call_op("gelu_and_mul", None, x, y)
        assert m._called_ops["gelu_and_mul"] == "reference.torch"
    print("  dispatch: flagos->default.flagos / reference->reference.torch PASS")
    print("  （kernel 精度/性能直测已移至 kernel 层）")
    print("  => A2 op-level PASS")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    name = sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"
    run(load_profile(name))
