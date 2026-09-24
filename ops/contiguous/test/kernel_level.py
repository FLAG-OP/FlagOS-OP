# kernel 层直测: 精度 / 别名 / 哨兵 / 性能。
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

    from kernel.torch_level import contiguous_torch
    from kernel.triton_level import contiguous_triton
    from reference import contiguous_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    max_err = 0.0

    # 1) 别名快路径: 已连续返回 self 本身
    for dt in dts:
        x = torch.randn(64, 128, dtype=dt, device=dev)
        assert contiguous_triton(x) is x, f"{dt} 连续输入未返回别名"

    # 2) 拷贝路径: 非连续（transpose/permute/slice/expand）→ 连续
    #    注意: 先按目标 dtype 造张量再取视图，避免 .to() 把 layout 归一成连续
    view_cases = [
        (torch.float32,
         lambda dt: torch.randn(256, 128, dtype=dt, device=dev).t()),
        (torch.bfloat16,
         lambda dt: torch.randn(8, 32, 16, 16, dtype=dt,
                                device=dev).permute(0, 2, 3, 1)),
        (torch.float16,
         lambda dt: torch.randn(512, 512, dtype=dt, device=dev)[::2, 1::3]),
        (torch.float32,
         lambda dt: torch.randn(1, 256, dtype=dt, device=dev).expand(64, 256)),
    ]
    for dt, mk in view_cases:
        x = mk(dt)
        assert not x.is_contiguous(), f"用例构造错误: {dt} 输入本应非连续"
        got, exp = contiguous_triton(x), contiguous_reference(x)
        assert got.is_contiguous(), f"{dt} 输出非连续"
        assert got.data_ptr() != x.data_ptr(), f"{dt} 未分配新存储"
        assert got.stride() == exp.stride(), \
            f"{dt} stride {got.stride()} != {exp.stride()}"
        err = (got.float() - exp.float()).abs().max().item()
        assert err < tol[dt], f"{dt} err={err}"
        max_err = max(max_err, err)

    # 2b) 二维非整倍数维度
    for shape in [(1, 14336), (8, 5000), (128, 5120), (7, 4097)]:
        for dt in dts:
            x = (torch.randn(shape[1], shape[0], dtype=dt, device=dev)).t()
            got, exp = contiguous_triton(x), contiguous_reference(x)
            assert got.is_contiguous() and got.stride() == exp.stride()
            err = (got.float() - exp.float()).abs().max().item()
            assert err < tol[dt], f"{shape} {dt} err={err}"
            max_err = max(max_err, err)

    # 3) 哨兵: 确定性 + 输入敏感
    x = torch.randn(256, 128, dtype=torch.float32, device=dev).t()
    o1 = contiguous_triton(x)
    assert torch.equal(o1, contiguous_triton(x)), "同输入两次调用不一致"
    assert not torch.equal(o1, contiguous_triton(x + 1)), "输出不随输入变化"

    # 4) 三方一致: triton vs torch 级
    x = torch.randn(512, 512, dtype=torch.float16, device=dev).t()
    assert torch.equal(contiguous_triton(x), contiguous_torch(x))

    # 5) 性能: 8192² 转置 → 连续
    x = torch.randn(8192, 8192, dtype=torch.float32, device=dev).t()
    t_ms = _bench(lambda: contiguous_triton(x), dev)
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
                            device=p.torch_device).t()
            return lambda: fn(x)
        return _make

    def bw(t):
        return {"GBps": 8192 * 8192 * 4 * 2 / t / 1e6}

    from kernel.triton_level import contiguous_triton
    return [PerfCase("ops.contiguous.triton", group="ops", level="kernel",
                     make_fn=make(contiguous_triton), derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
