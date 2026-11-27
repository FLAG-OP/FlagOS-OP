# kernel 层直测: 精度 / 哨兵 / 性能（issue #55 交付要求 3）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from reference import layer_norm_reference  # noqa: E402


def _abserr(got, exp):
    return (got.float() - exp.float()).abs().max().item()


def _sync(dev):
    import torch
    if isinstance(dev, str) and dev.startswith("cuda"):
        torch.cuda.synchronize(torch.device(dev))


def _bench(fn, dev, warmup=20, iters=100):
    import time
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1000


def run(profile):
    import torch

    from kernel.thead_ppu import layer_norm_ppu
    from kernel.torch_level import layer_norm_torch

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    tol = {torch.float32: 2e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    worst = 0.0

    # 1) 精度: 多形态 × 3 dtype（含非整倍数归一化维 5000/5120/4097/1023）
    cases = [
        ((16, 128), (128,), True, True),
        ((16, 5000), (5000,), True, True),
        ((8, 5120), (5120,), True, False),
        ((4, 7, 4097), (4097,), False, False),
        ((8, 16, 5120), (16, 5120), True, True),
    ]
    for shape, nshape, has_w, has_b in cases:
        for dt in dts:
            x = torch.randn(*shape, dtype=torch.float32, device=dev).to(dt)
            # weight/bias 形状 = normalized_shape 本身（2D 归一化维时是
            # 多维张量, ATen 语义）; ×0.1 缩放为典型 LN 分布（1 bf16 ulp
            # 守 1e-2 判卷线）
            w = (torch.randn(nshape, dtype=torch.float32,
                             device=dev).to(dt) * 0.1 if has_w else None)
            b = (torch.randn(nshape, dtype=torch.float32,
                             device=dev).to(dt) * 0.1 if has_b else None)
            got = layer_norm_ppu(x, nshape, w, b, 1e-5)
            exp = layer_norm_reference(x, nshape, w, b, 1e-5)
            err = _abserr(got, exp)
            assert err < tol[dt], f"{shape}/{nshape} {dt} err={err}"
            worst = max(worst, err)
            got_t = layer_norm_torch(x, nshape, w, b, 1e-5)
            assert _abserr(got_t, exp) < tol[dt], f"torch 级 {shape} {dt}"

    # 2) 哨兵: 确定性 + 输入敏感
    x = torch.randn(32, 1024, dtype=torch.float16, device=dev)
    w = torch.randn(1024, dtype=torch.float16, device=dev)
    o1 = layer_norm_ppu(x, (1024,), w, w, 1e-5)
    assert torch.equal(o1, layer_norm_ppu(x, (1024,), w, w, 1e-5))
    assert not torch.equal(o1, layer_norm_ppu(x + 1, (1024,), w, w, 1e-5)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 梯度链路: 原生 backward 节点
    xg = torch.randn(16, 512, dtype=torch.float32, device=dev,
                     requires_grad=True)
    wg = torch.randn(512, dtype=torch.float32, device=dev, requires_grad=True)
    layer_norm_ppu(xg, (512,), wg, None, 1e-5).sum().backward()
    assert xg.grad is not None and wg.grad is not None
    assert torch.isfinite(xg.grad).all()

    # 4) 性能: 短采样, Transformer 典型形态
    x = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    w = torch.randn(4096, dtype=torch.float16, device=dev)
    b = torch.randn(4096, dtype=torch.float16, device=dev)
    t_ms = _bench(lambda: layer_norm_ppu(x, (4096,), w, b, 1e-5), dev)
    bytes_ = x.numel() * x.element_size() * 2
    native_t = _bench(
        lambda: torch.nn.functional.layer_norm(x, (4096,), w, b, 1e-5), dev)

    return {"ok": True, "max_err": round(worst, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "native_latency_ms": round(native_t, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(4096, 4096, dtype=torch.float16,
                        device=p.torch_device)
        w = torch.randn(4096, dtype=torch.float16, device=p.torch_device)
        b = torch.randn(4096, dtype=torch.float16, device=p.torch_device)
        from kernel.thead_ppu import layer_norm_ppu
        return lambda: layer_norm_ppu(x, (4096,), w, b, 1e-5)

    def bw(t):
        return {"GBps": 4096 * 4096 * 2 * 2 / t / 1e6}

    return [PerfCase("ops.layer_norm.native", group="ops", level="kernel",
                     make_fn=make, derived=bw)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
