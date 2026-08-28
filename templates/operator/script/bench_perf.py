#!/usr/bin/env python3
"""性能测试（单算子独立版）: 短采样 + 同步 + 带宽换算。

注意: 多用例回归请挂 common/perf_registry.py 走 perf_run（每用例
独立子进程，规避分配器污染 known-issues #10）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))         # 仓库根（用 common.perf）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.perf import bench          # noqa: E402
from gen_golden import _load_reference  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shape", type=int, nargs=2, default=[8192, 8192])
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    fn = _load_reference()                 # <TODO: 换成候选实现>
    dt = getattr(torch, args.dtype)
    x = torch.randn(*args.shape, dtype=dt, device=args.device) * 2
    g = torch.randn(*args.shape, dtype=dt, device=args.device)
    t = bench(lambda: fn(x, g), warmup=20, iters=100, device=args.device)
    bytes_ = 3 * x.numel() * x.element_size()
    print(f"reference @ {args.shape} {args.dtype}: {t:.3f}ms "
          f"({bytes_ / t / 1e6:.0f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
