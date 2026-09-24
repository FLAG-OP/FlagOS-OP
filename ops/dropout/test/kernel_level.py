# kernel 层直测: 确定性分支 / 随机分支结构统计 / 哨兵 / 性能。
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


def _stochastic_ok(torch, inp, out, p, seed, fn, ratio_tol=0.02):
    """值域/缩放/比例/seed 可控 的结构+统计校验。"""
    reasons = []
    if out.shape != inp.shape or out.dtype != inp.dtype:
        reasons.append("shape/dtype")
    if out.data_ptr() == inp.data_ptr():
        reasons.append("别名")
    scale = 1.0 / (1.0 - p)
    keep = out != 0
    if keep.any():
        got = out[keep].float()
        want = inp[keep].float() * scale
        if (((got - want).abs() / (want.abs() + 1e-6)).max().item() > 0.05):
            reasons.append("缩放")
    drop = (~keep).float().mean().item()
    if abs(drop - p) > ratio_tol:
        reasons.append(f"drop={drop:.4f}")
    torch.manual_seed(seed)
    a = fn(inp, p, True)
    torch.manual_seed(seed)
    b = fn(inp, p, True)
    if not torch.equal(a, b):
        reasons.append("同seed不确定")
    torch.manual_seed(seed + 1)
    c = fn(inp, p, True)
    if torch.equal(a, c):
        reasons.append("seed不敏感")
    return (not reasons), reasons


def run(profile):
    import torch

    from kernel.torch_level import dropout_torch
    from kernel.triton_level import dropout_triton
    from reference import dropout_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)

    # 1) 确定性分支
    for dt in dts:
        x = torch.randn(64, 128, dtype=dt, device=dev)
        assert dropout_triton(x, 0.0, True) is x, "p=0 未返回别名"
        assert dropout_triton(x, 0.5, False) is x, "train=False 未返回别名"
        z = dropout_triton(x, 1.0, True)
        assert torch.equal(z, torch.zeros_like(x)), "p=1 非全零"

    # 1b) 非法 p
    for bad in (-0.1, 1.5):
        try:
            dropout_triton(torch.randn(8, device=dev), bad, True)
            raise AssertionError(f"p={bad} 未报错")
        except RuntimeError:
            pass

    # 2) 随机分支: 结构+统计
    max_drop_dev = 0.0
    for shape in [(64, 1024), (8, 5000), (128, 5120), (7, 4097)]:
        for dt in dts:
            for p in (0.1, 0.5, 0.9):
                inp = torch.randn(*shape, dtype=dt, device=dev) * 2
                seed = 1234
                torch.manual_seed(seed)
                out = dropout_triton(inp, p, True)
                ok, why = _stochastic_ok(torch, inp, out, p, seed,
                                         dropout_triton)
                assert ok, f"{shape} {dt} p={p}: {why}"
                drop = (out == 0).float().mean().item()
                max_drop_dev = max(max_drop_dev, abs(drop - p))

    # 3) channels_last 内存格式
    cl = torch.randn(2, 3, 16, 16, dtype=torch.float32, device=dev).to(
        memory_format=torch.channels_last)
    out = dropout_triton(cl, 0.5, True)
    assert out.stride() == cl.stride(), "channels_last stride 未保持"

    # 4) 三方一致（确定性分支位级；随机分支仅结构）
    x = torch.randn(512, 512, dtype=torch.float16, device=dev)
    assert dropout_triton(x, 0.0, True) is x
    assert torch.equal(dropout_triton(x, 1.0, True), dropout_torch(x, 1.0, True))

    # 5) 性能: 8192² fp16 p=0.5
    x = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
    t_ms = _bench(lambda: dropout_triton(x, 0.5, True), dev)
    bytes_ = 2 * x.numel() * x.element_size()
    return {"ok": True, "max_drop_dev": round(max_drop_dev, 4),
            "sentinel": "seed-deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.float16,
                            device=p.torch_device)
            return lambda: fn(x, 0.5, True)
        return _make

    def bw(t):
        return {"GBps": 8192 * 8192 * 2 * 2 / t / 1e6}

    from kernel.triton_level import dropout_triton
    return [PerfCase("ops.dropout.triton", group="ops", level="kernel",
                     make_fn=make(dropout_triton), derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
