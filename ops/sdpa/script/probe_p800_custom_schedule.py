#!/usr/bin/env python3
"""Validate and benchmark the experimental fixed-schedule P800 Triton kernel."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _consume(output):
    # XMLIR completion guard: a device-to-host read must follow every launch.
    return output[0, 0, 0, 0].item()


def _bench(call, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000)
    return sorted(samples)[len(samples) // 2]


def _make(B, Hq, Hkv, Sq, Skv, D, dev, dtype=None, seed=1):
    import torch

    dtype = dtype or torch.float16
    torch.manual_seed(seed)
    q = (torch.randn(B, Hq, Sq, D) * 0.03).to(dtype).to(dev)
    k = (torch.randn(B, Hkv, Skv, D) * 0.03).to(dtype).to(dev)
    v = (torch.randn(B, Hkv, Skv, D) * 0.03).to(dtype).to(dev)
    return q, k, v


def _accuracy_case(dev, B, Hq, Hkv, Sq, Skv, D, causal):
    import torch

    from kernel.p800_custom_triton import sdpa_p800_custom_triton
    from kernel.triton_level import sdpa_triton

    q, k, v = _make(B, Hq, Hkv, Sq, Skv, D, dev)
    gqa = Hq != Hkv
    ref = sdpa_triton(q, k, v, None, 0.0, causal, None, gqa)
    out = sdpa_p800_custom_triton(
        q, k, v, None, 0.0, causal, None, gqa
    )
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    return {
        "case": f"B{B}_H{Hq}-{Hkv}_S{Sq}-{Skv}_D{D}_c{int(causal)}",
        "max_err_vs_vendor": float(f"{err:.6g}"),
        "pass": bool(err < (2e-2 if q.dtype != torch.float32 else 1e-5)),
    }


def _mask_probe(dev):
    """Retain explicit evidence for the remaining mask launch blocker."""
    import torch

    from kernel.p800_custom_triton import _launch
    from kernel.triton_level import sdpa_triton

    results = []
    for kind in ("bool", "float"):
        q, k, v = _make(1, 4, 4, 128, 128, 64, dev, seed=4)
        if kind == "bool":
            mask = torch.rand(1, 4, 128, 128, device=dev) > 0.3
            bias = torch.zeros_like(mask, dtype=q.dtype)
            bias.masked_fill_(~mask, float("-inf"))
        else:
            mask = torch.randn(
                1, 4, 128, 128, device=dev, dtype=q.dtype
            ) * 0.1
            bias = mask.contiguous()
        try:
            out = _launch(
                q, k, v, bias, 0.0, False, None, False
            )
            torch.cuda.synchronize()
            ref = sdpa_triton(q, k, v, mask, 0.0, False, None, False)
            err = (out.float() - ref.float()).abs().max().item()
            item = {"mask": kind, "pass": bool(err < 2e-2),
                    "max_err_vs_vendor": float(f"{err:.6g}")}
        except Exception as exc:
            item = {
                "mask": kind,
                "pass": False,
                "error": f"{type(exc).__name__}: {str(exc)[:220]}",
            }
        results.append(item)
    return results


def _perf_case(dev, B, H, S, D, warmup, iters):
    from kernel.p800_custom_triton import sdpa_p800_custom_triton
    from kernel.triton_level import sdpa_triton

    q, k, v = _make(B, H, H, S, S, D, dev)

    def vendor():
        return _consume(sdpa_triton(q, k, v, None, 0.0, True, None, False))

    def custom():
        return _consume(sdpa_p800_custom_triton(
            q, k, v, None, 0.0, True, None, False
        ))

    vendor_ms = _bench(vendor, warmup, iters)
    custom_ms = _bench(custom, warmup, iters)
    ref = sdpa_triton(q, k, v, None, 0.0, True, None, False).float().cpu()
    out = sdpa_p800_custom_triton(
        q, k, v, None, 0.0, True, None, False
    ).float().cpu()
    err = (out - ref).abs().max().item()
    flops = 0.5 * 4 * B * H * S * S * D
    return {
        "B": B,
        "H": H,
        "S": S,
        "D": D,
        "vendor_ms": round(vendor_ms, 4),
        "custom_ms": round(custom_ms, 4),
        "vendor_tflops": round(flops / (vendor_ms / 1000) / 1e12, 1),
        "custom_tflops": round(flops / (custom_ms / 1000) / 1e12, 1),
        "vendor_over_custom": round(vendor_ms / custom_ms, 3),
        "max_err_vs_vendor": float(f"{err:.6g}"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    accuracy = [
        _accuracy_case(args.device, 1, 4, 4, 128, 128, 128, True),
        _accuracy_case(args.device, 1, 32, 8, 1024, 1024, 128, True),
        _accuracy_case(args.device, 1, 4, 4, 333, 333, 64, True),
        _accuracy_case(args.device, 1, 8, 2, 257, 257, 64, False),
        _accuracy_case(args.device, 1, 4, 4, 1, 7, 64, False),
    ]
    performance = [
        _perf_case(args.device, 1, 16, 1024, 64,
                   args.warmup, args.iters),
        _perf_case(args.device, 1, 16, 1024, 128,
                   args.warmup, args.iters),
        _perf_case(args.device, 1, 16, 4096, 128,
                   args.warmup, args.iters),
        _perf_case(args.device, 16, 16, 1024, 128,
                   args.warmup, args.iters),
    ]
    # Keep known launch failures last to avoid polluting timing above.
    masks = _mask_probe(args.device)
    result = {
        "device": args.device,
        "schedule": {
            "BLOCK_M": 64,
            "BLOCK_N": "min(64, HEAD_DIM)",
            "num_warps": 4,
            "num_stages": 1,
            "autotune": False,
        },
        "accuracy": accuracy,
        "performance": performance,
        "masks": masks,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
