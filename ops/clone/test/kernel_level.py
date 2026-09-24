# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics 供 run.py 落盘）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def _sync(dev):
    import torch
    if isinstance(dev, str) and dev.startswith("mlu"):
        import torch_mlu  # noqa: F401
        torch.mlu.synchronize()
    elif isinstance(dev, str) and dev.startswith("cuda"):
        torch.cuda.synchronize()


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

    from kernel.torch_level import clone_torch
    from kernel.triton_level import clone_triton
    from reference import clone_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    max_err = 0.0

    # 1) 精度 + 存储独立 + layout(shape/dtype/stride)
    for shape in [(64, 1024), (1, 14336), (8, 5000), (128, 5120), (7, 4097)]:
        for dt in dts:
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            got = clone_triton(x)
            exp = clone_reference(x)
            assert got.shape == exp.shape and got.dtype == dt
            assert got.stride() == exp.stride(), \
                f"{dt} stride {got.stride()} != {exp.stride()}"
            assert got.data_ptr() != x.data_ptr(), "clone 与 src 共享存储"
            err = (got.float() - exp.float()).abs().max().item()
            assert err < tol[dt], f"{shape} {dt} err={err}"
            max_err = max(max_err, err)
            # 修改 clone 不影响 src
            if shape == (64, 1024):
                got.add_(1.0)
                assert not torch.equal(got, x), "修改 clone 影响了 src"

    # 1b) 非 dense 视图 / memory_format
    view_cases = [
        (torch.float32, lambda: torch.randn(256, 128, device=dev).t()),
        (torch.bfloat16,
         lambda: torch.randn(8, 32, 16, 16, device=dev).permute(0, 2, 3, 1)),
        (torch.float16, lambda: torch.randn(512, 512, device=dev)[::2, 1::3]),
        (torch.float32,
         lambda: torch.randn(1, 256, device=dev).expand(64, 256)),
        (torch.float32,
         lambda: torch.randn(2, 3, 8, 8, device=dev).to(
             memory_format=torch.channels_last)),
    ]
    for dt, mk in view_cases:
        x = mk().to(dt)
        got, exp = clone_triton(x), clone_reference(x)
        assert got.stride() == exp.stride(), \
            f"view {dt} stride {got.stride()} != {exp.stride()}"
        err = (got.float() - exp.float()).abs().max().item()
        assert err < tol[dt], f"view {dt} err={err}"
        max_err = max(max_err, err)

    # 2) 哨兵: 确定性 + 输入敏感
    x = torch.randn(64, 1024, dtype=torch.float32, device=dev)
    o1 = clone_triton(x)
    assert torch.equal(o1, clone_triton(x)), "同输入两次调用不一致"
    assert not torch.equal(o1, clone_triton(x + 1)), "输出不随输入变化"

    # 3) 三方一致: triton vs torch 级（torch 级=empty_like+copy_）
    x = torch.randn(2048, 2048, dtype=torch.float16, device=dev)
    assert torch.equal(clone_triton(x), clone_torch(x)), "triton 与 torch 级不一致"

    # 4) 性能: 8192² fp32
    x = torch.randn(8192, 8192, dtype=torch.float32, device=dev)
    t_ms = _bench(lambda: clone_triton(x), dev)
    bytes_ = 2 * x.numel() * x.element_size()
    return {"ok": True, "max_err": round(max_err, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.float32,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    def bw(t):
        return {"GBps": 8192 * 8192 * 4 * 2 / t / 1e6}

    from kernel.triton_level import clone_triton
    return [PerfCase("ops.clone.triton", group="ops", level="kernel",
                     make_fn=make(clone_triton), derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
