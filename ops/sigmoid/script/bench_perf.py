#!/usr/bin/env python3
"""sigmoid 独立性能测试：原生、torch 级和 Triton 级。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from common.perf import bench  # noqa: E402

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shape", type=int, nargs="+", default=[2048, 2048])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--impl",
                        choices=["native", "torch", "triton", "flaggems"],
                        default="triton")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true",
                        help="最后一行输出可供 bench_matrix 解析的 JSON")
    args = parser.parse_args()
    if args.impl == "native":
        candidate = torch.sigmoid
    elif args.impl == "torch":
        from ops.sigmoid.kernel.torch_level import sigmoid_torch
        candidate = sigmoid_torch
    elif args.impl == "triton":
        from ops.sigmoid.kernel.triton_level import sigmoid_triton
        candidate = sigmoid_triton
    else:
        import flag_gems
        candidate = flag_gems.sigmoid
    x = torch.randn(*args.shape, dtype=getattr(torch, args.dtype),
                    device=args.device)
    candidate(x)
    latency = bench(lambda: candidate(x), warmup=args.warmup,
                    iters=args.iters, device=args.device)
    bytes_moved = 2 * x.numel() * x.element_size()
    result = {
        "impl": args.impl,
        "shape": args.shape,
        "numel": x.numel(),
        "dtype": args.dtype,
        "latency_ms": round(latency, 6),
        "GBps": round(bytes_moved / latency / 1e6, 2),
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"{args.impl} @ {tuple(args.shape)} {args.dtype}: "
              f"{latency:.6f} ms, {result['GBps']:.2f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
