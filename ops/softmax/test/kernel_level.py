# kernel 层直测: 精度 / 哨兵 / 性能（issue #55 交付要求 3）。
#
# 覆盖: 多 dim（尾维/中间维/负维）× 3 dtype × dtype 参数（含
# half_to_float 语义）× 非整倍数尾块（3072/5000/5120, #15a 教训——
# softmax 是行归约, 尾块维度必测）; 大数值输入（×8, 验证减 max 稳定）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from reference import softmax_reference  # noqa: E402


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

    from kernel.thead_ppu import softmax_ppu

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    tol = {torch.float32: 2e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    worst = 0.0

    # 1) 精度: 形状 × dim × dtype（含非整倍数行宽 5000/5120/1023）
    for shape, dim in [((32, 512), -1), ((8, 5000), -1), ((128, 5120), -1),
                       ((4, 8, 5120), -1), ((7, 1023), -1),
                       ((8, 16, 5120), 1), ((4, 64, 128), 2)]:
        for dt in dts:
            x = (torch.randn(*shape, dtype=torch.float32,
                             device=dev) * 8).to(dt)
            got = softmax_ppu(x, dim)
            exp = softmax_reference(x, dim)
            assert got.shape == exp.shape, f"{shape} dim={dim} shape"
            err = _abserr(got, exp)
            assert err < tol[dt], f"softmax {shape} dim={dim} {dt} err={err}"
            worst = max(worst, err)
            # 行和守恒（归一化语义自检）
            s = got.float().sum(dim=dim)
            assert (s - 1).abs().max().item() < 2e-2, f"{shape} 行和≠1"

    # 1b) dtype 参数语义: fp16 输入 + dtype=fp32 → 输出 fp32（ATen 语义）
    x = torch.randn(32, 512, dtype=torch.float16, device=dev) * 8
    got = softmax_ppu(x, -1, torch.float32)
    exp = softmax_reference(x, -1, torch.float32)
    assert got.dtype == torch.float32, f"dtype 参数输出 {got.dtype}"
    err = _abserr(got, exp)
    assert err < 2e-5, f"half_to_float err={err}"
    worst = max(worst, err)

    # 2) 哨兵: 确定性 + 输入敏感
    x = torch.randn(64, 1024, dtype=torch.float16, device=dev) * 8
    o1 = softmax_ppu(x, -1)
    assert torch.equal(o1, softmax_ppu(x, -1)), "同输入两次调用不一致"
    assert not torch.equal(o1, softmax_ppu(-x, -1)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 性能: 短采样, attention scores 典型形态
    x = torch.randn(16, 32, 4096, 4096, dtype=torch.float16, device=dev) * 8
    t_ms = _bench(lambda: softmax_ppu(x, -1), dev)
    bytes_ = x.numel() * x.element_size() * 2
    native_t = _bench(lambda: torch.softmax(x, -1), dev)

    return {"ok": True, "max_err": round(worst, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "native_latency_ms": round(native_t, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(16, 32, 4096, 4096, dtype=torch.float16,
                        device=p.torch_device) * 8
        from kernel.thead_ppu import softmax_ppu
        return lambda: softmax_ppu(x, -1)

    def bw(t):
        return {"GBps": 16 * 32 * 4096 * 4096 * 2 * 2 / t / 1e6}

    return [PerfCase("ops.softmax.native", group="ops", level="kernel",
                     make_fn=make, derived=bw)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
