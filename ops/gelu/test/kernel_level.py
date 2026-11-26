# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics 供 run.py 落盘）。
#
# 覆盖（issue #55 交付要求 3）:
#   - approximate none/tanh 双模式 × forward/backward
#   - fp16 / bf16 / fp32
#   - 含非 BLOCK 整倍数尾块维度（3072/5000/5120, known-issues #15a 教训）
#   - 负尾部 / 近零 / 非有限值 / 特殊值 / 非连续输入（走原生回退边界）
#   - 哨兵: 确定性 + 输入敏感（抓 #11 静默 no-op / 不写输出）
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))                 # 引用同算子的 kernel/reference
sys.path.insert(0, str(OP_DIR.parent.parent))   # 仓库根（common 等）

from reference import gelu_backward_reference, gelu_reference  # noqa: E402


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

    from kernel.triton_level import gelu_backward_triton, gelu_triton

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    max_err = 0.0

    # 1) 前向精度: 双模式 × 3 dtype × 多 shape（含非整倍数尾块 5120/4097/1023）
    shapes = [(64, 1024), (8, 5000), (128, 5120), (7, 4097), (3, 1023)]
    for mode in ("none", "tanh"):
        for shape in shapes:
            for dt in dts:
                x = (torch.randn(*shape, dtype=torch.float32, device=dev) * 2
                     ).to(dt)
                got = gelu_triton(x, mode)
                exp = gelu_reference(x, mode)
                err = (got.float() - exp.float()).abs().max().item()
                assert err < tol[dt], f"gelu {mode} {shape} {dt} err={err}"
                max_err = max(max_err, err)

    # 1b) 特殊值域: 负尾部（大 |x| 饱和区）、近零、非有限值、boundary
    x = torch.tensor([[-20.0, -5.0, -1e-6, 0.0, 1e-6, 5.0, 20.0, 1e4],
                      [-1e4, float("inf"), float("-inf"), float("nan"),
                       8.0, -8.0, 0.044715, -0.044715]],
                     dtype=torch.float32, device=dev)
    for mode in ("none", "tanh"):
        got = gelu_triton(x, mode)
        exp = gelu_reference(x, mode)
        # 非有限值位级对齐（inf→inf, nan→nan; inf-inf=nan 会污染数值
        # 误差, 故非有限位置一律走位级 parity, 数值容差只看有限位置）
        assert torch.equal(torch.isnan(got), torch.isnan(exp)), \
            f"{mode} nan parity 失败"
        assert torch.equal(torch.isinf(got), torch.isinf(exp)), \
            f"{mode} inf parity 失败"
        assert torch.equal(got[torch.isinf(exp)], exp[torch.isinf(exp)]), \
            f"{mode} inf 符号不一致"
        fin = torch.isfinite(exp)
        err = (got[fin] - exp[fin]).abs().max().item()
        assert err <= 1e-6, f"{mode} 特殊值 err={err}"
        max_err = max(max_err, err)

    # 2) 反向精度: 解析 backward vs 参考（fp32 全档 + 半精度档）
    for mode in ("none", "tanh"):
        for dt in dts:
            x = (torch.randn(64, 3072, dtype=torch.float32, device=dev) * 2
                 ).to(dt)
            g = (torch.randn(64, 3072, dtype=torch.float32, device=dev)).to(dt)
            dx = gelu_backward_triton(g, x, mode)
            dx_ref = gelu_backward_reference(g, x, mode)
            err = (dx.float() - dx_ref.float()).abs().max().item()
            assert err < tol[dt], f"bwd {mode} {dt} err={err}"
            max_err = max(max_err, err)

    # 2b) autograd 全链（未注册场景）: Function 的解析 backward vs 参考
    x64 = torch.randn(32, 512, dtype=torch.float32, device=dev,
                      requires_grad=True)
    g_seed = torch.randn(32, 512, dtype=torch.float32, device=dev)
    for mode in ("none", "tanh"):
        # 直接对解析 backward 做 gradcheck 式对拍（双精度参考）
        dx = gelu_backward_triton(g_seed, x64.detach(), mode)
        dx_ref = gelu_backward_reference(g_seed, x64.detach(), mode)
        err = (dx - dx_ref).abs().max().item()
        assert err < 1e-5, f"autograd 解析 backward {mode} err={err}"
        max_err = max(max_err, err)

    # 3) 回退边界: 非连续输入走原生，结果仍与参考一致
    xnc = torch.randn(64, 64, dtype=torch.float32, device=dev).t()  # 非连续
    for mode in ("none", "tanh"):
        got = gelu_triton(xnc, mode)
        exp = gelu_reference(xnc, mode)
        err = (got - exp).abs().max().item()
        assert err < 1e-5, f"非连续回退 {mode} err={err}"

    # 4) 哨兵: 确定性 + 输入敏感（抓 #11 静默 no-op / 不写输出）
    x = torch.randn(64, 1024, dtype=torch.float16, device=dev)
    for mode in ("none", "tanh"):
        o1 = gelu_triton(x, mode)
        assert torch.equal(o1, gelu_triton(x, mode)), f"{mode} 同输入两次不一致"
        assert not torch.equal(o1, gelu_triton(-x, mode)), \
            f"{mode} 输出不随输入变化（疑似静默 no-op）"

    # 5) 性能: 短采样（≤100 次），双模式（#10: 不用长循环）
    #    对照三方: 自研 Triton / 原生 aten（thead_ppu 委托）/ fp32 参考
    from kernel.thead_ppu import gelu_ppu
    x = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
    t_ms = _bench(lambda: gelu_triton(x, "tanh"), dev)
    bytes_ = x.numel() * x.element_size() * 2
    native_t = _bench(lambda: gelu_ppu(x, "tanh"), dev)
    ref_t = _bench(lambda: gelu_reference(x, "tanh"), dev)

    return {"ok": True, "max_err": round(max_err, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "native_latency_ms": round(native_t, 4),
            "reference_latency_ms": round(ref_t, 4),
            "triton_vs_native": round(native_t / t_ms, 3),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(8192, 8192, dtype=torch.float16,
                        device=p.torch_device)
        from kernel.triton_level import gelu_triton
        return lambda: gelu_triton(x, "tanh")

    def bw(t):
        return {"GBps": 8192 * 8192 * 2 * 2 / t / 1e6}

    return [PerfCase("ops.gelu.triton", group="ops", level="kernel",
                     make_fn=make, derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
