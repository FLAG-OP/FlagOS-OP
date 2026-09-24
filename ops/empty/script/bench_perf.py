#!/usr/bin/env python3
"""性能测试（empty）: 分配开销采样。"""
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
    ap.add_argument("--impl", default="torch", help="reference | torch")
    args = ap.parse_args()
    fn = _load(*{
        "reference": ("reference.py", "empty_reference"),
        "torch": ("kernel/torch_level.py", "empty_torch"),
    }[args.impl])
    dt = getattr(torch, args.dtype)
    t = bench(lambda: fn(tuple(args.shape), dtype=dt, device=args.device),
              warmup=20, iters=100, device=args.device)
    print(f"{args.impl} empty {tuple(args.shape)} {args.dtype}: {t:.4f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
