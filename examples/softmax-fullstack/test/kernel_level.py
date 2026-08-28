# 算子库层直测: 精度 / 哨兵 / 性能（与样板 test/kernel_level.py 同构）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def _bench(fn, iters=100, warm=20):
    import time

    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def run(profile):
    import torch
    import torch.nn.functional as F

    from kernel.triton_level import softmax_triton

    dev = profile.torch_device
    max_err = 0.0

    # 精度: 多 shape×dtype，含非整倍数 N（known-issues #15 尾块回归）
    for shape in [(128, 512), (256, 1024), (64, 2048), (1, 4096), (512, 256),
                  (32, 3072), (8, 5000), (16, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 3
            ref = F.softmax(x.float(), dim=-1).to(dt)
            out = softmax_triton(x)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            err = (out.float() - ref.float()).abs().max().item()
            assert err < tol, f"{shape} {dt} err={err}"
            max_err = max(max_err, err)

    # 哨兵
    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev)
    o1 = softmax_triton(x)
    assert torch.equal(o1, softmax_triton(x)) and \
        not torch.equal(o1, softmax_triton(x + 1))

    # 性能: 自研 / 原生 / FlagGems 三方
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
    t_tri = _bench(lambda: softmax_triton(x))
    t_ref = _bench(lambda: F.softmax(x, dim=-1))
    bw = x.numel() * 2 * 2 / t_tri / 1e6
    t_fg = None
    try:
        from flag_gems import ops as FG
        t_fg = _bench(lambda: FG.softmax(x, -1))
    except Exception:
        pass

    return {"ok": True, "max_err": round(max_err, 8),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_tri, 4), "GBps": round(bw, 1),
            "torch_ms": round(t_ref, 4),
            "flaggems_ms": round(t_fg, 4) if t_fg else None,
            "config": "BLOCK_N=2048 fixed (#15b)"}
