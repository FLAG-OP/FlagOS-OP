#!/usr/bin/env python3
"""Measure native, direct Triton, and A1 matmul in isolated processes."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _run_mode(args):
    import torch

    from common.device import load_profile

    profile = load_profile(args.device)
    m, k, n = args.shape
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(20260930)
    a = torch.randn(m, k, generator=generator).to(profile.torch_device, dtype) * 0.25
    b = torch.randn(k, n, generator=generator).to(profile.torch_device, dtype) * 0.25
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    native = torch.matmul(a, b).float()
    counter = {}

    if args._mode == "native":
        fn = torch.matmul
    elif args._mode == "direct":
        from kernel.triton_level import matmul_triton

        fn = matmul_triton
    else:
        from register import dispatch_key_for_profile, register_a1

        lib = register_a1(dispatch_key_for_profile(profile), counter=counter)
        fn = torch.matmul

    output = fn(a, b).float()
    if args._mode == "a1" and counter.get("forward", 0) != 1:
        raise AssertionError("A1 matmul was not intercepted")
    error = ((output - native).abs().max().item()
             / max(native.abs().max().item(), 1e-6))
    if error > 5e-2:
        raise AssertionError(f"{args._mode} relative error {error} exceeds 5e-2")

    def call():
        return fn(a, b).reshape(-1)[0].item()

    for _ in range(args.warmup):
        call()
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    if args._mode == "a1" and counter.get("forward", 0) != 1 + args.warmup + args.iters:
        raise AssertionError("A1 matmul interception count changed during timing")
    assert args._mode != "a1" or lib is not None
    return {
        "mode": args._mode,
        "median_ms": round(statistics.median(samples), 6),
        "p20_ms": round(samples[args.iters // 5], 6),
        "p80_ms": round(samples[(args.iters * 4) // 5], 6),
        "relative_error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--shape", nargs=3, type=int, default=[63, 127, 65],
                    metavar=("M", "K", "N"))
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                    default="bfloat16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--_mode", choices=("native", "direct", "a1"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if any(dim <= 0 for dim in args.shape) or args.warmup < 0 or not 1 <= args.iters <= 100:
        ap.error("shape must be positive, warmup nonnegative, and iters in 1..100")

    if args._mode:
        print(json.dumps(_run_mode(args)))
        return 0

    rows = []
    for mode in ("native", "direct", "a1"):
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--device", args.device, "--dtype", args.dtype,
                   "--shape", *(str(dim) for dim in args.shape),
                   "--warmup", str(args.warmup), "--iters", str(args.iters),
                   "--_mode", mode]
        child = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
        if child.returncode:
            sys.stderr.write(child.stdout + child.stderr)
            return child.returncode
        row = json.loads(child.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"{mode:7s} median={row['median_ms']:.6f} ms "
              f"p20={row['p20_ms']:.6f} p80={row['p80_ms']:.6f}", flush=True)

    by_mode = {row["mode"]: row for row in rows}
    result = {
        "device": args.device,
        "shape": args.shape,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "completion_guard": "reshape(-1)[0].item() per call",
        "isolation": "one process per implementation",
        "rows": rows,
        "a1_minus_direct_ms": round(
            by_mode["a1"]["median_ms"] - by_mode["direct"]["median_ms"], 6),
    }
    print(f"A1 minus direct: {result['a1_minus_direct_ms']:.6f} ms")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"saved {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
