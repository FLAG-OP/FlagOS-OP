#!/usr/bin/env python3
"""A1 kernel 层: Triton gelu kernel 直测（不经任何注册/分发）。"""
from __future__ import annotations


def run(profile) -> bool:
    import time

    import torch
    from routes.a1_aten import register_aten as RA

    dev = profile.torch_device
    print("=" * 60)
    print(f"A1 kernel [{profile.name}]: Triton gelu_tanh 直测 @ {dev}")
    print("=" * 60)

    K = RA._load_kernels()

    # ---- 精度 vs CPU 参考 ----
    for shape in [(4096, 4096), (8192, 2048), (128, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            ref = torch.nn.functional.gelu(x.cpu(), approximate="tanh").to(dev)
            out = K.gelu_tanh_triton(x)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            assert (out.float() - ref.float()).abs().max().item() < tol
    print("  精度: 9/9 组合 PASS")

    # ---- 哨兵: 确定性 & 输入敏感 ----
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    o1 = K.gelu_tanh_triton(x)
    assert torch.equal(o1, K.gelu_tanh_triton(x)), "同输入两次调用不一致"
    assert not torch.equal(o1, K.gelu_tanh_triton(x + 1.0)), "输出不随输入变化"
    print("  哨兵: 确定性 OK 输入敏感 OK")

    # ---- 性能（短采样，规避分配器池增长失真） ----
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    for _ in range(20):
        K.gelu_tanh_triton(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        K.gelu_tanh_triton(x)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    bw = x.numel() * 2 * 2 / t / 1e9 * 1e3
    print(f"  性能: {t:.3f}ms ({bw:.0f} GB/s)")
    print("  => A1 kernel PASS")
    return True


def perf_cases(profile):
    """性能回归用例（scripts/perf_run.py 消费），与 run() 同口径。"""
    from common.perf import PerfCase
    from routes.a1_aten import register_aten as RA

    K = RA._load_kernels()

    def make(p):
        import torch
        x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=p.torch_device) * 2
        return lambda: K.gelu_tanh_triton(x)

    def bw(t):
        return {"GBps": 8192 * 8192 * 2 * 2 / t / 1e6}

    return [PerfCase("matrix-kernel.a1.gelu_tanh.triton", group="matrix-kernel",
                     level="kernel", make_fn=make, derived=bw)]
