#!/usr/bin/env python3
"""测量 sigmoid 直调 kernel 与 A1 aten 路径的分发开销。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _bench(fn, device: str, warmup: int = 20, iters: int = 100) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="p800-kunlunxin")
    args = parser.parse_args()

    import torch
    from common.device import load_profile
    from ops.sigmoid.kernel.triton_level import sigmoid_triton
    import importlib
    register = importlib.import_module("ops.sigmoid.register")

    profile = load_profile(args.device)
    x = torch.randn(64, 1024, dtype=torch.float32,
                    device=profile.torch_device)
    direct_us = _bench(lambda: sigmoid_triton(x), profile.torch_device)
    register.register_a1(profile.dispatch_key, platform=profile.name)
    aten_us = _bench(lambda: torch.sigmoid(x), profile.torch_device)
    overhead_us = aten_us - direct_us
    print(f"A1 sigmoid: direct={direct_us:.3f} us, "
          f"registered={aten_us:.3f} us, overhead={overhead_us:.3f} us "
          f"({overhead_us / direct_us * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
