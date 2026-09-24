# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics 供 run.py 落盘）。
#
# type_as 是纯 elementwise cast（无归约），#15a 尾块归约坑不适用，但仍
# 覆盖非 BLOCK 整倍数维度；#11 device 上下文在 kernel/triton_level.py 内。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))                 # 引用同算子的 kernel/reference
sys.path.insert(0, str(OP_DIR.parent.parent))   # 仓库根（common 等）


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

    from kernel.torch_level import type_as_torch
    from kernel.triton_level import type_as_triton
    from reference import type_as_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    max_err = 0.0

    # 1) 精度: 多 shape（含非 BLOCK 整倍数 14336/5000/5120/4097）
    #    × 多 self/target dtype 组合 vs 参考
    for shape in [(64, 1024), (1, 14336), (8, 5000), (128, 5120), (7, 4097)]:
        for sdt in dts:
            for tdt in dts:
                if sdt == tdt:
                    continue
                x = torch.randn(*shape, dtype=sdt, device=dev) * 2
                other = torch.empty(4, dtype=tdt, device=dev)
                got = type_as_triton(x, other)
                exp = type_as_reference(x, other)
                assert got.dtype == tdt, f"{sdt}->{tdt} dtype {got.dtype}"
                assert got.stride() == exp.stride(), \
                    f"{sdt}->{tdt} stride {got.stride()} != {exp.stride()}"
                err = (got.float() - exp.float()).abs().max().item()
                assert err < tol[tdt], f"{shape} {sdt}->{tdt} err={err}"
                max_err = max(max_err, err)

    # 1b) 非 dense 视图: 转置 / permute / 步长切片 / expand / channels_last
    view_cases = [
        (torch.float32, torch.float16,
         lambda: torch.randn(256, 128, device=dev).t()),
        (torch.bfloat16, torch.float32,
         lambda: torch.randn(8, 32, 16, 16, device=dev).permute(0, 2, 3, 1)),
        (torch.float16, torch.bfloat16,
         lambda: torch.randn(512, 512, device=dev)[::2, 1::3]),
        (torch.float32, torch.bfloat16,
         lambda: torch.randn(1, 256, device=dev).expand(64, 256)),
        (torch.float32, torch.float16,
         lambda: torch.randn(2, 3, 8, 8, device=dev).to(
             memory_format=torch.channels_last)),
    ]
    for sdt, tdt, mk in view_cases:
        x = mk().to(sdt)
        other = torch.empty(4, dtype=tdt, device=dev)
        got = type_as_triton(x, other)
        exp = type_as_reference(x, other)
        assert got.stride() == exp.stride(), \
            f"view {sdt}->{tdt} stride {got.stride()} != {exp.stride()}"
        err = (got.float() - exp.float()).abs().max().item()
        assert err < tol[tdt], f"view {sdt}->{tdt} err={err}"
        max_err = max(max_err, err)

    # 1c) 同 dtype 别名快路径: 返回 self 本身（与 ATen 一致）
    x = torch.randn(64, 64, dtype=torch.bfloat16, device=dev)
    assert type_as_triton(x, x) is x, "same-dtype 未返回别名"

    # 1d) 不支持 dtype 走原生回退（int32），仍与参考一致
    xi = torch.randint(-1000, 1000, (256, 256), dtype=torch.int32, device=dev)
    oi = torch.empty(4, dtype=torch.float32, device=dev)
    assert torch.equal(type_as_triton(xi, oi).float(),
                       type_as_reference(xi, oi).float()), "int32 回退不一致"

    # 2) 哨兵: 确定性 + 输入敏感（抓静默 no-op / 未初始化输出）
    x = torch.randn(64, 1024, dtype=torch.float16, device=dev)
    other = torch.empty(4, dtype=torch.float32, device=dev)
    o1 = type_as_triton(x, other)
    assert torch.equal(o1, type_as_triton(x, other)), "同输入两次调用不一致"
    assert not torch.equal(o1, type_as_triton(x + 1, other)), \
        "输出不随输入变化（疑似静默 no-op）"

    # 3) 三方一致性: triton vs torch 级 vs reference
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    other = torch.empty(4, dtype=torch.float32, device=dev)
    assert torch.equal(type_as_triton(x, other), type_as_torch(x, other)), \
        "triton 与 torch 级不一致"

    # 4) 性能: 短采样（≤100 次），输出逐次新分配（#10: 不用长循环）
    x = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
    other = torch.empty(4, dtype=torch.float32, device=dev)
    t_ms = _bench(lambda: type_as_triton(x, other), dev)
    bytes_ = x.numel() * x.element_size() + x.numel() * other.element_size()

    return {"ok": True, "max_err": round(max_err, 10),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4),
            "GBps": round(bytes_ / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(8192, 8192, dtype=torch.float16,
                        device=p.torch_device)
        other = torch.empty(4, dtype=torch.float32, device=p.torch_device)
        from kernel.triton_level import type_as_triton
        return lambda: type_as_triton(x, other)

    def bw(t):
        return {"GBps": 8192 * 8192 * (2 + 4) / t / 1e6}

    return [PerfCase("ops.type_as.triton", group="ops", level="kernel",
                     make_fn=make, derived=bw)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
