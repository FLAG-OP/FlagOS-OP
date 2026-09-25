#!/usr/bin/env python3
"""Benchmark embedding forward/backward against native implementations.

Runs on either platform: pass ``--device cuda:1`` (P800/XMLIR) or
``--device mlu:0`` (Cambricon); the backend is picked by device type.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

FORWARD_SHAPES = [
    # (tag, num_weights, num_indices, dim)
    ("small_1k_d128", 4096, 1024, 128),
    ("prefill_16k_d128", 4096, 16384, 128),
    ("long_131k_d128", 4096, 131072, 128),
    ("wide_16k_d512", 4096, 16384, 512),
    ("vocab128k_16k_d128", 128256, 16384, 128),
]


def _consume(output):
    return output.reshape(-1)[0].item()


def _bench(call, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000)
    return sorted(samples)[len(samples) // 2]


def main() -> int:
    import torch

    from kernel.platform import embedding, embedding_backward

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                    default="float16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    dev = args.device

    rows = []
    print(
        f"{'shape':22s} {'ours(ms)':>10s} {'native(ms)':>11s} "
        f"{'triton(ms)':>11s} {'speedup':>9s}"
    )
    print("-" * 70)
    for tag, num_weights, num_indices, dim in FORWARD_SHAPES:
        torch.manual_seed(1000 + num_indices + dim)
        weight = (
            torch.randn(num_weights, dim) * 0.1
        ).to(dtype).to(dev)
        indices = torch.randint(
            0, num_weights, (num_indices,), device=dev
        )

        ours_ms = _bench(
            lambda: _consume(embedding(weight, indices)),
            args.warmup, args.iters,
        )
        native_ms = _bench(
            lambda: _consume(torch.ops.aten.embedding(
                weight, indices, -1, False, False
            )),
            args.warmup, args.iters,
        )

        try:
            from kernel.triton_level import embedding_triton
            triton_ms = _bench(
                lambda: _consume(embedding_triton(weight, indices)),
                1, min(args.iters, 5),
            )
        except Exception as exc:
            print(f"    [triton fail @ {tag}] {type(exc).__name__}: {exc}")
            triton_ms = float("nan")

        row = {
            "shape": tag,
            "num_weights": num_weights,
            "num_indices": num_indices,
            "dim": dim,
            "dtype": args.dtype,
            "ours_ms": round(ours_ms, 4),
            "native_ms": round(native_ms, 4),
            "triton_ms": round(triton_ms, 4)
            if triton_ms == triton_ms else None,
            "speedup_vs_native": round(native_ms / ours_ms, 3),
        }
        rows.append(row)
        print(
            f"{tag:22s} {ours_ms:10.4f} {native_ms:11.4f} "
            f"{triton_ms:11.4f} {native_ms / ours_ms:8.3f}x"
        )

    # Backward comparison. Native may not implement scale_grad_by_freq
    # (XPU raises ``Check scale_grad_by_freq == false failed``; torch_mlu
    # supports it) -> try native per platform and skip on failure.
    bwd_rows = []
    num_weights, num_indices, dim = 4096, 16384, 128
    torch.manual_seed(4321)
    weight = (torch.randn(num_weights, dim) * 0.1).to(dtype).to(dev)
    indices = torch.randint(0, num_weights, (num_indices,), device=dev)
    grad = (torch.randn(num_indices, dim) * 0.1).to(dtype).to(dev)
    for scale in (False, True):
        ours_ms = _bench(
            lambda: _consume(embedding_backward(
                grad, indices, num_weights, -1, scale, False
            )),
            args.warmup, args.iters,
        )
        item = {
            "shape": "bwd_16k_d128",
            "scale_grad_by_freq": scale,
            "ours_ms": round(ours_ms, 4),
        }
        try:
            native_ms = _bench(
                lambda: _consume(torch.ops.aten.embedding_backward(
                    grad, indices, num_weights, -1, scale, False
                )),
                args.warmup, args.iters,
            )
        except RuntimeError as exc:  # native 缺口（XPU）如实记为 null
            native_ms = None
            item["native_error"] = str(exc).splitlines()[0][:80]
        if native_ms is not None:
            item["native_ms"] = round(native_ms, 4)
            item["speedup_vs_native"] = round(native_ms / ours_ms, 3)
        bwd_rows.append(item)
        print("backward", item, flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "device": dev,
            "dtype": args.dtype,
            "completion_guard": "reshape(-1)[0].item()",
            "forward": rows,
            "backward": bwd_rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
