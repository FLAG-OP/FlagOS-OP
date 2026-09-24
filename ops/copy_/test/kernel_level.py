# kernel 层直测: 精度（同形/广播/cast）/ 原地返回 / 哨兵 / 性能。
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

    from kernel.torch_level import copy_torch
    from kernel.triton_level import copy_triton
    from reference import copy_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    max_err = 0.0

    # 1) 同形 + cast: dst 初始随机，src 异 dtype
    for shape in [(64, 1024), (1, 14336), (8, 5000), (7, 4097)]:
        for dd in dts:
            for sd in dts:
                dst = torch.randn(*shape, dtype=dd, device=dev)
                src = torch.randn(*shape, dtype=sd, device=dev) * 2
                ref = dst.clone()
                copy_reference(ref, src)
                out = copy_triton(dst, src)
                assert out is dst, "copy_triton 未返回 dst 本身"
                assert out.dtype == dd, f"dtype {out.dtype} != {dd}"
                err = (out.float() - ref.float()).abs().max().item()
                assert err < tol[dd], f"{shape} {dd}<-{sd} err={err}"
                max_err = max(max_err, err)

    # 2) 广播
    for dshape, sshape, dd in [((64, 128, 256), (128, 1), torch.float32),
                               ((8, 1, 5000), (8, 1, 1), torch.bfloat16),
                               ((16, 32), (1, 32), torch.float16)]:
        dst = torch.randn(*dshape, dtype=dd, device=dev)
        src = torch.randn(*sshape, dtype=dd, device=dev) * 2
        ref = dst.clone()
        copy_reference(ref, src)
        out = copy_triton(dst, src)
        assert out is dst
        err = (out.float() - ref.float()).abs().max().item()
        assert err < tol[dd], f"broadcast {dshape}<-{sshape} err={err}"
        max_err = max(max_err, err)

    # 3) 非连续 dst / src
    dst = torch.randn(512, 512, dtype=torch.float32, device=dev)[::2, 1::3]
    src = torch.randn(256, 171, dtype=torch.float32, device=dev)
    ref = dst.clone()
    copy_reference(ref, src)
    out = copy_triton(dst, src)
    assert out is dst
    err = (out.float() - ref.float()).abs().max().item()
    assert err < tol[torch.float32], f"noncontig err={err}"
    max_err = max(max_err, err)

    # 4) 自拷贝（别名）走回退，不崩且不变
    a = torch.randn(256, 256, dtype=torch.float32, device=dev)
    b = a.clone()
    out = copy_triton(a, a)
    assert out is a and torch.equal(a, b), "自拷贝语义错误"

    # 5) 哨兵: 确定性 + 输入敏感
    dst = torch.zeros(64, 1024, dtype=torch.float16, device=dev)
    src = torch.randn(64, 1024, dtype=torch.float16, device=dev)
    copy_triton(dst, src)
    o1 = dst.clone()
    dst2 = torch.zeros_like(dst)
    copy_triton(dst2, src)
    assert torch.equal(o1, dst2), "同输入两次调用不一致"
    dst3 = torch.zeros_like(dst)
    copy_triton(dst3, src + 1)
    assert not torch.equal(o1, dst3), "输出不随输入变化"

    # 6) 三方一致
    dst = torch.zeros(512, 512, dtype=torch.float16, device=dev)
    src = torch.randn(512, 512, dtype=torch.float16, device=dev)
    assert torch.equal(copy_triton(dst.clone(), src), copy_torch(dst.clone(), src))

    # 7) 性能: 8192² 同形原地
    dst = torch.empty(8192, 8192, dtype=torch.float32, device=dev)
    src = torch.randn(8192, 8192, dtype=torch.float32, device=dev)
    t_ms = _bench(lambda: copy_triton(dst, src), dev)
    bytes_ = 2 * src.numel() * src.element_size()
    return {"ok": True, "max_err": round(max_err, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            dst = torch.empty(8192, 8192, dtype=torch.float32,
                              device=p.torch_device)
            src = torch.randn(8192, 8192, dtype=torch.float32,
                              device=p.torch_device)
            return lambda: fn(dst, src)
        return _make

    def bw(t):
        return {"GBps": 8192 * 8192 * 4 * 2 / t / 1e6}

    from kernel.triton_level import copy_triton
    return [PerfCase("ops.copy_.triton", group="ops", level="kernel",
                     make_fn=make(copy_triton), derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
