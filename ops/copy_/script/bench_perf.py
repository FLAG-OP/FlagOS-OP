#!/usr/bin/env python3
"""性能测试（copy_ 单算子）: 同形原地拷贝，短采样 + 同步 + 带宽换算。

用法:
  python3 script/bench_perf.py --device mlu --dtype float32 --shape 8192 8192
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.perf import bench          # noqa: E402


def _load(rel: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        rel.replace("/", "_")[:-3], HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mlu")
    ap.add_argument("--shape", type=int, nargs=2, default=[8192, 8192])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--impl", default="triton",
                    help="reference | torch | triton")
    args = ap.parse_args()

    IMPLS = {
        "reference": ("reference.py", "copy_reference"),
        "torch": ("kernel/torch_level.py", "copy_torch"),
        "triton": ("kernel/triton_level.py", "copy_triton"),
    }
    fn = _load(*IMPLS[args.impl])
    dt = getattr(torch, args.dtype)
    dst = torch.empty(*args.shape, dtype=dt, device=args.device)
    src = torch.randn(*args.shape, dtype=dt, device=args.device)
    t = bench(lambda: fn(dst, src), warmup=20, iters=100, device=args.device)
    bytes_ = 2 * src.numel() * src.element_size()
    print(f"{args.impl} @ {tuple(args.shape)} {args.dtype}: {t:.3f}ms "
          f"({bytes_ / t / 1e6:.0f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
