# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics 供 run.py 落盘）。
#
# 覆盖（issue #55 交付要求 3）:
#   - 2D/3D 批维 × 有/无 bias × fp16/bf16/fp32
#   - 非整倍数特征维（5000/5120/4097）+ 方形/瘦高/扁宽 GEMM 形态
#   - GEMM 判卷: 相对误差 5e-2 rel（docs/acceptance.md 唯一权威表）
#   - 哨兵: 确定性 + 输入敏感
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


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

    from kernel.thead_ppu import linear_ppu
    from kernel.torch_level import linear_torch
    from reference import linear_reference

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    worst = 0.0

    # 1) 精度: (2D/3D) × (bias/no-bias) × 3 dtype × 多 shape
    #    （含非整倍数 in/out 特征维; GEMM 用 5e-2 rel 判卷）
    shapes = [  # (batch..., M, K, N)
        ((64, 512), 128),      # 2D 基础
        ((4, 8, 5000), 5120),  # 3D + 非整倍数特征维
        ((7, 4097), 1024),     # 非整倍数 M
        ((1, 1023), 128),      # 瘦 M
        ((8, 128), 4096),      # 扁宽 K
    ]
    for bshape, n_out in shapes:
        k_in = bshape[-1]
        for dt in dts:
            x = (torch.randn(*bshape, dtype=torch.float32, device=dev)).to(dt)
            w = (torch.randn(n_out, k_in, dtype=torch.float32,
                             device=dev) * 0.05).to(dt)
            for bias_on in (True, False):
                b = (torch.randn(n_out, dtype=torch.float32,
                                 device=dev) * 0.05).to(dt) if bias_on else None
                got = linear_ppu(x, w, b)
                exp = linear_reference(x, w, b)
                assert got.shape == exp.shape, f"{bshape}->{n_out} shape"
                err = _relerr(got, exp)
                assert err < 5e-2, f"linear {bshape}->{n_out} {dt} rel={err}"
                worst = max(worst, err)
                # torch 级对照同容差
                got_t = linear_torch(x, w, b)
                err_t = _relerr(got_t, exp)
                assert err_t < 5e-2, f"torch 级 {bshape} {dt} rel={err_t}"

    # 2) 哨兵: 确定性 + 输入敏感
    x = torch.randn(64, 256, dtype=torch.float16, device=dev)
    w = torch.randn(128, 256, dtype=torch.float16, device=dev) * 0.05
    o1 = linear_ppu(x, w)
    assert torch.equal(o1, linear_ppu(x, w)), "同输入两次调用不一致"
    assert not torch.equal(o1, linear_ppu(x + 1, w)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 梯度链路: autograd 经原生节点（注册后场景在 op 层验证）
    xg = torch.randn(32, 128, dtype=torch.float32, device=dev,
                     requires_grad=True)
    wg = torch.randn(64, 128, dtype=torch.float32, device=dev,
                     requires_grad=True)
    y = linear_ppu(xg, wg)
    y.sum().backward()
    assert xg.grad is not None and wg.grad is not None, "梯度缺失"
    assert torch.isfinite(xg.grad).all() and torch.isfinite(wg.grad).all()

    # 4) 性能: 短采样（≤100 次），典型 Transformer linear 形态
    x = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
    w = torch.randn(4096, 4096, dtype=torch.float16, device=dev) * 0.02
    t_ms = _bench(lambda: linear_ppu(x, w), dev)
    flops = 2 * 4096 * 4096 * 4096
    native_t = _bench(lambda: torch.nn.functional.linear(x, w), dev)

    return {"ok": True, "max_rel_err": round(worst, 6),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "native_latency_ms": round(native_t, 4),
            "TFLOPS": round(flops / t_ms / 1e9, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(4096, 4096, dtype=torch.float16,
                        device=p.torch_device)
        w = torch.randn(4096, 4096, dtype=torch.float16,
                        device=p.torch_device) * 0.02
        from kernel.thead_ppu import linear_ppu
        return lambda: linear_ppu(x, w)

    def tflops(t):
        return {"TFLOPS": 2 * 4096 ** 3 / t / 1e9}

    return [PerfCase("ops.linear.native", group="ops", level="kernel",
                     make_fn=make, derived=tflops)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
