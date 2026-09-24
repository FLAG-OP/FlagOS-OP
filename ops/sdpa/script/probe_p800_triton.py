#!/usr/bin/env python3
"""Probe FlagGems' Kunlunxin Triton SDPA without autotune interference.

The experiment calls the underlying JIT kernel with deterministic tile configs,
compares it with the P800 vendor-delegate backend, and consumes one output
element on every timed call. Consuming output is mandatory on XMLIR: discarding
the result can measure only launch/submission rather than execution.
"""
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
    # A device-to-host read queued after the kernel forces completion while
    # avoiding a full-output reduction kernel in the measured interval.
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


def _run_case(dev: str, B: int, H: int, S: int, D: int, cfgs,
              warmup: int, iters: int):
    import torch
    import triton
    from flag_gems.runtime import torch_device_fn
    from flag_gems.runtime.backend._kunlunxin.ops.attention import _attn_fwd

    from common.xpu_compat import ensure_xpu_compiler_debuggable
    from kernel.triton_level import sdpa_triton

    ensure_xpu_compiler_debuggable()
    torch.manual_seed(1)
    dtype = torch.float16
    q = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(dev)
    k = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(dev)
    v = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(dev)
    out = torch.empty_like(q)
    lse = torch.empty((B, H, S), device=dev, dtype=torch.float32)
    scale = D ** -0.5

    def vendor_call():
        return _consume(sdpa_triton(q, k, v, None, 0.0, True, None, False))

    vendor_ms = _bench(vendor_call, warmup, iters)
    vendor_result = sdpa_triton(
        q, k, v, None, 0.0, True, None, False
    ).float().cpu()
    row = {
        "B": B,
        "H": H,
        "S": S,
        "D": D,
        "vendor_ms": round(vendor_ms, 4),
        "configs": [],
    }

    for block_m, block_n, num_warps, num_stages in cfgs:
        grid = (triton.cdiv(S, block_m), B * H, 1)

        def triton_call():
            with torch_device_fn.device(dev):
                _attn_fwd.fn[grid](
                    q,
                    k,
                    v,
                    None,
                    scale,
                    lse,
                    out,
                    *q.stride(),
                    *k.stride(),
                    *v.stride(),
                    1,
                    1,
                    1,
                    1,
                    *out.stride(),
                    B,
                    H,
                    H,
                    1,
                    S,
                    S,
                    D,
                    STAGE=3,
                    HAS_ATTN_MASK=False,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    PRE_LOAD_V=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            return _consume(out)

        try:
            triton_ms = _bench(triton_call, warmup, iters)
            err = (out.float().cpu() - vendor_result).abs().max().item()
            item = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "ms": round(triton_ms, 4),
                "max_err_vs_vendor": float(f"{err:.6g}"),
                "vendor_over_triton": round(vendor_ms / triton_ms, 3),
            }
        except Exception as exc:  # Experimental probe: keep scanning configs.
            item = {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            }
        row["configs"].append(item)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--mode", choices=["micro", "fa2"], default="micro")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    # Deterministic configs chosen from the 2026-09-22 scan. XMLIR timing was
    # insensitive to num_stages in the observed set; keep one stage for stability.
    cfgs_d64 = [(32, 32, 4, 1), (64, 32, 4, 1), (64, 64, 4, 1)]
    cfgs_d128 = [(64, 64, 4, 1), (128, 64, 8, 1), (64, 32, 4, 1)]
    if args.mode == "micro":
        cases = [
            (1, 16, 1024, 64, cfgs_d64),
            (1, 16, 1024, 128, cfgs_d128),
            (1, 16, 4096, 128, cfgs_d128),
        ]
    else:
        cases = [
            (16, 32, 1024, 64, cfgs_d64),
            (16, 16, 1024, 128, cfgs_d128),
            (4, 16, 4096, 128, cfgs_d128),
        ]

    rows = []
    for B, H, S, D, cfgs in cases:
        print(f"probe B={B} H={H} S={S} D={D}", flush=True)
        row = _run_case(args.device, B, H, S, D, cfgs,
                        args.warmup, args.iters)
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "device": args.device,
            "mode": args.mode,
            "completion_guard": "output[0,0,0,0].item()",
            "rows": rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
