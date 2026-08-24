#!/usr/bin/env python3
"""A2 算子层测试: 精度 / dispatch / 性能 三段式"""
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

    # ---- stage 1: accuracy ----
    for shape in [(4096, 4096), (8192, 2048), (1, 14336), (128, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            y = torch.randn(*shape, dtype=dt, device=dev)
            ref = K.gelu_and_mul_reference(x, y)
            out = K.gelu_and_mul_triton(x, y)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            assert (out.float() - ref.float()).abs().max().item() < tol
    print("  stage 1/3 accuracy : 12/12 PASS")

    # ---- stage 2: dispatch ----
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
    print("  stage 2/3 dispatch : PASS (flagos/reference 切换正确)")

    # ---- stage 3: benchmark ----
    shape = (8192, 8192)
    x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    y = torch.randn(*shape, dtype=torch.bfloat16, device=dev)

    def bench(fn, iters=200, warmup=20):
        for _ in range(warmup):
            fn(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x, y)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_tri = bench(K.gelu_and_mul_triton)
    t_ref = bench(K.gelu_and_mul_reference)
    bw = x.numel() * 2 * 3 / t_tri / 1e9 * 1e3
    print(f"  stage 3/3 bench    : Triton={t_tri:.3f}ms ({bw:.0f} GB/s) "
          f"ref={t_ref:.3f}ms speedup={t_ref/t_tri:.1f}x")
    print("  => A2 op-level PASS")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    name = sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"
    run(load_profile(name))
