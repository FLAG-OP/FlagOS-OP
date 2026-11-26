# kernel 层直测: 精度 / 哨兵 / 性能（issue #55 交付要求 3）。
#
# 覆盖: 2D/3D/广播批维/向量形态 × 3 dtype; GEMM 5e-2 rel 判卷;
# 非整倍数 K/M/N（5000/5120/4097）; 哨兵: 确定性 + 输入敏感。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from reference import matmul_reference  # noqa: E402


def _relerr(got, exp):
    denom = exp.float().abs().clamp_min(1.0)
    return ((got.float() - exp.float()).abs() / denom).max().item()


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

    from kernel.thead_ppu import matmul_ppu

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    worst = 0.0

    # 1) 精度: 多形态 × 3 dtype（GEMM 5e-2 rel 判卷）
    cases = [  # (self_shape, other_shape)
        ((64, 128), (128, 96)),          # 2D 基础
        ((64, 5000), (5000, 512)),       # 非整倍数 K
        ((5120, 128), (128, 4097)),      # 非整倍数 M/N
        ((1, 128), (128, 96)),           # 向量-矩阵（1×K）
        ((64, 128), (128, 1)),           # 矩阵-向量
        ((4, 64, 128), (128, 96)),       # 3D @ 2D 广播
        ((4, 64, 128), (4, 128, 96)),    # 3D @ 3D
        ((4, 1, 64, 128), (2, 128, 96)),  # 批维广播
    ]
    for sshape, oshape in cases:
        for dt in dts:
            a = (torch.randn(*sshape, dtype=torch.float32,
                             device=dev) * 0.5).to(dt)
            b = (torch.randn(*oshape, dtype=torch.float32,
                             device=dev) * 0.5).to(dt)
            got = matmul_ppu(a, b)
            exp = matmul_reference(a, b).to(dev)
            assert got.shape == exp.shape, f"{sshape}@{oshape} shape"
            err = _relerr(got, exp)
            assert err < 5e-2, f"matmul {sshape}@{oshape} {dt} rel={err}"
            worst = max(worst, err)

    # 2) 哨兵: 确定性 + 输入敏感
    a = torch.randn(64, 128, dtype=torch.float16, device=dev)
    b = torch.randn(128, 96, dtype=torch.float16, device=dev)
    o1 = matmul_ppu(a, b)
    assert torch.equal(o1, matmul_ppu(a, b)), "同输入两次调用不一致"
    assert not torch.equal(o1, matmul_ppu(a + 1, b)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 梯度链路: 原生 autograd 节点（mm backward）
    ag = torch.randn(32, 64, dtype=torch.float32, device=dev,
                     requires_grad=True)
    bg = torch.randn(64, 32, dtype=torch.float32, device=dev,
                     requires_grad=True)
    matmul_ppu(ag, bg).sum().backward()
    assert ag.grad is not None and bg.grad is not None
    assert torch.isfinite(ag.grad).all()

    # 4) 性能: 短采样, Transformer 典型 GEMM（M=N=K=4096）
    a = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    t_ms = _bench(lambda: matmul_ppu(a, b), dev)
    flops = 2 * 4096 ** 3
    native_t = _bench(lambda: torch.matmul(a, b), dev)

    return {"ok": True, "max_rel_err": round(worst, 6),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "native_latency_ms": round(native_t, 4),
            "TFLOPS": round(flops / t_ms / 1e9, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        a = torch.randn(4096, 4096, dtype=torch.float16,
                        device=p.torch_device)
        b = torch.randn(4096, 4096, dtype=torch.float16,
                        device=p.torch_device)
        from kernel.thead_ppu import matmul_ppu
        return lambda: matmul_ppu(a, b)

    def tflops(t):
        return {"TFLOPS": 2 * 4096 ** 3 / t / 1e9}

    return [PerfCase("ops.matmul.native", group="ops", level="kernel",
                     make_fn=make, derived=tflops)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
