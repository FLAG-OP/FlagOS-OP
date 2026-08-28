#!/usr/bin/env python3
"""分发开销实测: 直调 kernel vs 框架分发路径（A1 aten / A2 call_op）。

用法:
  python3 scripts/bench_dispatch.py --route a1 --device p800-kunlunxin
  python3 scripts/bench_dispatch.py --route a2 --device p800-kunlunxin

注意: 本栈小 kernel 的"直调"耗时本身就有 ~50µs 启动开销 floor
（64×1024 与 8192×8192 直调同为 ~52µs），解读分发开销时计入基数。
A2 的 call_op 长循环在本栈偶发挂起，故默认短循环。
"""
from __future__ import annotations

import argparse
import time


def bench(fn, n=200, warm=20):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6   # µs


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", choices=["a1", "a2"], default="a1")
    ap.add_argument("--device", default="p800-kunlunxin")
    args = ap.parse_args()

    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))

    from common.device import load_profile
    profile = load_profile(args.device)
    dev = profile.torch_device

    import torch
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev) * 2
    y = torch.randn_like(x)

    if args.route == "a1":
        import torch.nn.functional as F
        from routes.a1_aten import register_aten as RA
        K = RA._load_kernels()
        direct = bench(lambda: K.gelu_tanh_triton(x))
        RA.register_gelu_aten(profile.dispatch_key)
        aten = bench(lambda: F.gelu(x, approximate="tanh"))
        print(f"A1 aten: 直调={direct:.1f}µs  注册后={aten:.1f}µs  "
              f"分发开销≈{aten-direct:.1f}µs "
              f"({(aten-direct)/direct*100:.0f}%)")
    else:
        import os
        from routes.a2_dispatch.plugin import kernels as RK
        os.environ["VLLM_FL_PLUGIN_MODULES"] = \
            "routes.a2_dispatch.plugin.register_ops"
        from vllm_fl.dispatch import (call_op, get_default_manager,
                                      reset_default_manager)
        from vllm_fl.dispatch.policy import with_preference
        reset_default_manager()
        get_default_manager().ensure_initialized()
        direct = bench(lambda: RK.gelu_and_mul_triton(x, y))
        with with_preference("flagos"):
            disp = bench(lambda: call_op("gelu_and_mul", None, x, y),
                         n=100, warm=10)
        print(f"A2 call_op: 直调={direct:.1f}µs  分发后={disp:.1f}µs  "
              f"开销≈{disp-direct:.1f}µs ({(disp-direct)/direct*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
