#!/usr/bin/env python3
"""A2 kernel 层: Triton gelu_and_mul kernel 直测（不经 dispatch）。"""
from __future__ import annotations


def run(profile):
    import time

    import torch
    from routes.a2_dispatch.plugin import kernels as K

    dev = profile.torch_device
    print("=" * 60)
    print(f"A2 kernel [{profile.name}]: Triton gelu_and_mul 直测 @ {dev}")
    print("=" * 60)

    # ---- 精度 vs PyTorch 语义参考 ----
    max_err = 0.0
    for shape in [(4096, 4096), (1, 14336), (128, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            y = torch.randn(*shape, dtype=dt, device=dev)
            ref = K.gelu_and_mul_reference(x, y)
            out = K.gelu_and_mul_triton(x, y)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            err = (out.float() - ref.float()).abs().max().item()
            assert err < tol                       # 容差随 dtype 不同，逐组断言
            max_err = max(max_err, err)
    print("  精度: 9/9 组合 PASS")

    # ---- 哨兵 ----
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    y = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    o1 = K.gelu_and_mul_triton(x, y)
    assert torch.equal(o1, K.gelu_and_mul_triton(x, y))
    assert not torch.equal(o1, K.gelu_and_mul_triton(x + 1.0, y))
    print("  哨兵: 确定性 OK 输入敏感 OK")

    # ---- 性能 ----
    shape = (8192, 8192)
    x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    y = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    for _ in range(20):
        K.gelu_and_mul_triton(x, y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        K.gelu_and_mul_triton(x, y)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    bw = x.numel() * 2 * 3 / t / 1e9 * 1e3
    print(f"  性能: {t:.3f}ms ({bw:.0f} GB/s)")
    print("  => A2 kernel PASS")
    return {"ok": True, "max_err": round(max_err, 8),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t, 4), "GBps": round(bw, 1)}


def perf_cases(profile):
    """性能回归用例（scripts/perf_run.py 消费），与 run() 同口径。"""
    from common.perf import PerfCase
    from routes.a2_dispatch.plugin import kernels as K

    def make(p):
        import torch
        x = torch.randn(8192, 8192, dtype=torch.bfloat16,
                        device=p.torch_device) * 2
        y = torch.randn(8192, 8192, dtype=torch.bfloat16, device=p.torch_device)
        return lambda: K.gelu_and_mul_triton(x, y)

    def bw(t):
        return {"GBps": 8192 * 8192 * 2 * 3 / t / 1e6}

    return [PerfCase("matrix-kernel.a2.gelu_and_mul.triton", group="matrix-kernel",
                     level="kernel", make_fn=make, derived=bw)]
