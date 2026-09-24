#!/usr/bin/env python3
"""性能测试（type_as 单算子独立版）: 短采样 + 同步 + 带宽换算。

注意: 多用例回归请挂 common/perf_registry.py 走 perf_run（每用例独立
子进程，规避分配器污染 known-issues #10）。

用法:
  python3 script/bench_perf.py --device mlu --dtype float16 \
      --target-dtype float32 --shape 8192 8192
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))         # 仓库根（用 common.perf）
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
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--target-dtype", default="float32")
    ap.add_argument("--impl", default="triton",
                    help="reference | torch | triton")
    args = ap.parse_args()

    IMPLS = {
        "reference": ("reference.py", "type_as_reference"),
        "torch": ("kernel/torch_level.py", "type_as_torch"),
        "triton": ("kernel/triton_level.py", "type_as_triton"),
    }
    fn = _load(*IMPLS[args.impl])

    dt = getattr(torch, args.dtype)
    tdt = getattr(torch, args.target_dtype)
    x = torch.randn(*args.shape, dtype=dt, device=args.device) * 2
    other = torch.empty(4, dtype=tdt, device=args.device)
    t = bench(lambda: fn(x, other), warmup=20, iters=100, device=args.device)
    bytes_ = x.numel() * x.element_size() + x.numel() * tdt.itemsize
    print(f"{args.impl} @ {tuple(args.shape)} {args.dtype}->"
          f"{args.target_dtype}: {t:.3f}ms ({bytes_ / t / 1e6:.0f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
