# kernel 层直测: _softmax 原语（half_to_float 双档）精度 / 哨兵 / 性能。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from reference import softmax_prim_reference  # noqa: E402


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

    from kernel.thead_ppu import _softmax_ppu

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    tol = {torch.float32: 2e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    worst = 0.0

    # 1) 精度: half_to_float=False（输出同 dtype）
    for shape, dim in [((32, 512), -1), ((8, 5000), -1), ((128, 5120), -1),
                       ((4, 8, 5120), -1), ((4, 64, 128), 2)]:
        for dt in dts:
            x = (torch.randn(*shape, dtype=torch.float32,
                             device=dev) * 8).to(dt)
            got = _softmax_ppu(x, dim, False)
            exp = softmax_prim_reference(x, dim, False)
            err = _abserr(got, exp)
            assert err < tol[dt], f"_softmax {shape} dim={dim} {dt} err={err}"
            worst = max(worst, err)

    # 1b) half_to_float=True: fp16/bf16 输入 → fp32 输出（ATen 语义）
    for dt in (torch.float16, torch.bfloat16):
        x = (torch.randn(32, 512, dtype=torch.float32, device=dev) * 8).to(dt)
        got = _softmax_ppu(x, -1, True)
        exp = softmax_prim_reference(x, -1, True)
        assert got.dtype == torch.float32, f"h2f 输出 {got.dtype}"
        err = _abserr(got, exp)
        assert err < 2e-5, f"h2f {dt} err={err}"
        worst = max(worst, err)
    # fp32 输入 + h2f=True（语义仍成立, 输出 fp32）
    x32 = torch.randn(32, 512, dtype=torch.float32, device=dev) * 8
    got = _softmax_ppu(x32, -1, True)
    assert _abserr(got, softmax_prim_reference(x32, -1, True)) < 2e-5

    # 2) 哨兵: 确定性 + 输入敏感
    x = torch.randn(64, 1024, dtype=torch.float16, device=dev) * 8
    o1 = _softmax_ppu(x, -1, False)
    assert torch.equal(o1, _softmax_ppu(x, -1, False))
    assert not torch.equal(o1, _softmax_ppu(-x, -1, False)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 性能: 短采样, attention scores 形态
    x = torch.randn(16, 32, 4096, 4096, dtype=torch.float16, device=dev) * 8
    t_ms = _bench(lambda: _softmax_ppu(x, -1, False), dev)
    bytes_ = x.numel() * x.element_size() * 2

    return {"ok": True, "max_err": round(worst, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(16, 32, 4096, 4096, dtype=torch.float16,
                        device=p.torch_device) * 8
        from kernel.thead_ppu import _softmax_ppu
        return lambda: _softmax_ppu(x, -1, False)

    def bw(t):
        return {"GBps": 16 * 32 * 4096 * 4096 * 2 * 2 / t / 1e6}

    return [PerfCase("ops._softmax.native", group="ops", level="kernel",
                     make_fn=make, derived=bw)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
