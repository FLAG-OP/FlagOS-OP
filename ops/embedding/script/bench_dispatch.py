#!/usr/bin/env python3
"""Measure native→A1 dispatch overhead for the embedding operator.

The default invocation launches separate subprocesses for native/direct/A1 so
process-wide ``AutogradCUDA`` registration cannot pollute the baseline.
"""
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


def _consume(output):
    # XMLIR completion guard. Launch-only timing is invalid on this stack.
    return output.reshape(-1)[0].item()


def _run_mode(mode: str, dev: str, warmup: int, iters: int) -> dict:
    import torch

    torch.manual_seed(20260924)
    weight = (
        torch.randn(4096, 128, device=dev, dtype=torch.float16) * 0.1
    )
    indices = torch.randint(0, 4096, (16384,), device=dev)

    if mode == "native":
        def call():
            return _consume(torch.ops.aten.embedding(
                weight, indices, -1, False, False
            ))
    elif mode == "direct":
        from kernel.p800_kunlunxin import embedding

        def call():
            return _consume(embedding(weight, indices))
    elif mode == "a1":
        from register import register_a1

        register_a1("AutogradCUDA")

        def call():
            return _consume(torch.nn.functional.embedding(
                indices, weight, padding_idx=-1,
                scale_grad_by_freq=False, sparse=False,
            ))
    else:  # argparse prevents this branch in normal use.
        raise ValueError(mode)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000)
    torch.cuda.synchronize()
    return {
        "mode": mode,
        "median_ms": round(statistics.median(samples), 6),
        "p20_ms": round(sorted(samples)[iters // 5], 6),
        "p80_ms": round(sorted(samples)[(iters * 4) // 5], 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--_mode", choices=("native", "direct", "a1"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._mode:
        print(json.dumps(_run_mode(
            args._mode, args.device, args.warmup, args.iters
        )))
        return 0

    rows = []
    for mode in ("native", "direct", "a1"):
        proc = subprocess.run(
            [
                sys.executable, __file__,
                "--device", args.device,
                "--warmup", str(args.warmup),
                "--iters", str(args.iters),
                "--_mode", mode,
            ],
            text=True, capture_output=True, cwd=str(ROOT),
        )
        if proc.returncode:
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
            return proc.returncode
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(row, flush=True)

    by_mode = {row["mode"]: row for row in rows}
    overhead = {
        "a1_minus_native_ms": round(
            by_mode["a1"]["median_ms"] - by_mode["native"]["median_ms"], 6
        ),
        "a1_minus_direct_ms": round(
            by_mode["a1"]["median_ms"] - by_mode["direct"]["median_ms"], 6
        ),
    }
    result = {
        "device": args.device,
        "shape": {
            "num_weights": 4096,
            "num_indices": 16384,
            "dim": 128,
            "dtype": "float16",
        },
        "completion_guard": "reshape(-1)[0].item()",
        "rows": rows,
        **overhead,
    }
    print(json.dumps(overhead, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
