#!/usr/bin/env python3
"""Compare matmul forward or backward latency with deterministic CPU inputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
INPUT_SEED = 20260929
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))


def _device(value):
    if value == "cpu" or ":" in value:
        return value
    from common.device import load_profile
    return load_profile(value).torch_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", required=True, help="profile name or torch device")
    ap.add_argument("--shape", nargs=4, type=int, default=[1, 128, 256, 192],
                    metavar=("B", "M", "K", "N"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--impl", default="all",
                    choices=("all", "triton", "native", "flaggems"))
    ap.add_argument("--direction", choices=("forward", "backward"), default="forward")
    ap.add_argument("--case", help="named nonempty golden case; overrides --shape")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.impl == "all":
        results = {}
        unavailable = {}
        for name in ("triton", "native", "flaggems"):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--device", args.device, "--dtype", args.dtype,
                       "--shape", *(str(dim) for dim in args.shape),
                       "--impl", name, "--direction", args.direction]
            if args.case:
                command.extend(["--case", args.case])
            child = subprocess.run(command, text=True, capture_output=True, timeout=180)
            if child.returncode:
                if name == "flaggems":
                    reason = (child.stderr.strip().splitlines() or
                              [f"exit code {child.returncode}"])[-1]
                    unavailable[name] = reason
                    print(f"flaggems: unavailable ({reason})")
                    continue
                sys.stderr.write(child.stdout + child.stderr)
                return child.returncode
            result = json.loads(child.stdout.strip().splitlines()[-1])
            if result is None:
                unavailable[name] = "no linalg_matmul or bmm entry"
                print("flaggems: unavailable")
                continue
            results[name] = result
            print(f"{name:9s} {result['latency_ms']:.4f} ms  "
                  f"{result['tflops']:.2f} TFLOPS  "
                  f"rel={result['relative_error']:.2e}")
        report = {"device": _device(args.device), "shape": None if args.case else args.shape,
                  "case": args.case, "direction": args.direction,
                  "dtype": args.dtype, "warmup": 20, "iters": 100,
                  "input_seed": next(iter(results.values()))["input_seed"],
                  "timing": "per-call scalar readback",
                  "isolation": "one process per implementation",
                  "results": results, "unavailable": unavailable}
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"saved {args.json_out}")
        return 0

    from common.perf import bench
    from ops.matmul.script.perf_common import prepare, workloads

    device = _device(args.device)
    batch, m, k, n = args.shape
    if args.case:
        match = next(((c, seed) for c, dt, seed in workloads()
                      if c["name"] == args.case and dt == args.dtype), None)
        if match is None:
            ap.error(f"unknown or empty performance case: {args.case}")
        case, seed = match
    else:
        if min(args.shape) <= 0:
            ap.error("performance shapes must be positive")
        case = {"name": "custom", "a_shape": [batch, m, k],
                "b_shape": [batch, k, n]}
        seed = INPUT_SEED
    fn, info = prepare(case, args.dtype, seed, args.impl, args.direction, device)
    latency = bench(fn, warmup=20, iters=100, device=device)
    tflops = info["flops"] / latency / 1e9
    print(json.dumps({"latency_ms": latency, "tflops": tflops,
                      "relative_error": info["max_check_error"], "input_seed": seed,
                      "entry": args.impl, "direction": args.direction,
                      "a_shape": case["a_shape"], "b_shape": case["b_shape"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
