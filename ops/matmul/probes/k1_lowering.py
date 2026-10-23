#!/usr/bin/env python3
"""Locate K=1 dot failures with controlled preceding kernels and saved IR.

Use a separate process and TRITON_CACHE_DIR for each XMLIR compiler setting.
Only launches on --device and writes to --output; never modifies the backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ops.matmul.kernel.triton_level import _matmul_kernel
from ops.matmul.probes.outer import _matmul_outer_kernel
from flag_gems.runtime import torch_device_fn


# Preserve the old specialization explicitly even after production is fixed.
_original_kernel = triton.jit(_matmul_kernel.fn, do_not_specialize=[])
_dynamic_k_kernel = _matmul_kernel


def launch(a, b, c, variant="original"):
    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        grid = (triton.cdiv(m, 32) * triton.cdiv(n, 32), 1)
        if variant == "outer":
            return _matmul_outer_kernel[grid](
                a, b, c, 0, *a.stride(), 0, *b.stride(), 0, *c.stride(),
                M=m, N=n, K=k, BM=32, BN=32, num_warps=4)
        kernel = _dynamic_k_kernel if variant == "dynamic" else _original_kernel
        return kernel[grid](
            a, b, c, m, n, k, 0, *a.stride(), 0, *b.stride(),
            0, *c.stride(), BM=32, BN=32, BK=32, num_warps=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16")
    ap.add_argument("--variant", choices=("original", "dynamic"), default="original")
    ap.add_argument("--prime-backend", choices=("original", "outer"), default="original")
    ap.add_argument("--prime-dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16")
    ap.add_argument("--prime-size", type=int, default=128)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    report = {"torch": torch.__version__, "triton": triton.__version__,
              "device": args.device, "dtype": args.dtype,
              "variant": args.variant, "prime_backend": args.prime_backend,
              "prime_dtype": args.prime_dtype, "prime_size": args.prime_size,
              "env": {k: os.environ.get(k) for k in (
                  "CUDA_VISIBLE_DEVICES", "XMLIR_MATMUL_FAST_MODE",
                  "TRITON_SDNN_SINGLE_CODE_MODE", "TRITON_CACHE_DIR")},
              "repeats": args.repeats, "runs": [], "artifacts": {}}
    # Allocate before the test sequence, so no intervening device op changes SRAM.
    inputs = {}
    for k in (1, 2, 32):
        for value in (0.0, 1.0, 3.0):
            a = torch.full((37, k), value, dtype=dtype, device=args.device)
            b = torch.full((k, 65), value, dtype=dtype, device=args.device)
            c = torch.empty((37, 65), dtype=dtype, device=args.device)
            inputs[k, value] = a, b, c
    primes = {}
    for value in (0.0, 1.0, 3.0):
        pdtype = getattr(torch, args.prime_dtype)
        primes[value] = (
            torch.full((args.prime_size, 32), value, dtype=pdtype, device=args.device),
            torch.full((32, args.prime_size), value, dtype=pdtype, device=args.device),
            torch.empty((args.prime_size, args.prime_size), dtype=pdtype, device=args.device),
        )
    for k in (1, 2, 32):
        compiled = launch(*inputs[k, 0.0], args.variant)
        artifacts = {}
        for stage in ("ttir", "ttxir", "llir", "xpubin"):
            content = compiled.asm[stage]
            raw = content.encode() if isinstance(content, str) else bytes(content)
            name = f"k{k}.{stage}"
            (args.output / name).write_bytes(raw)
            artifacts[stage] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}
        report["artifacts"][str(k)] = artifacts
    inputs[1, 0.0][2].cpu()  # Complete warmup before the controlled sequence.

    for prime_value in (0.0, 1.0, 3.0, 1.0, 0.0):
        for k in (1, 2):
            for target_value in (0.0, 1.0):
                for repeat in range(args.repeats):
                    launch(*primes[prime_value], args.prime_backend)
                    launch(*inputs[k, target_value], args.variant)
                    got = inputs[k, target_value][2].float().cpu()
                    want = k * target_value * target_value
                    error = (got - want).abs()
                    bad = ~torch.isfinite(got) | (error > 0.05 * max(want, 1.0))
                    worst = error.nan_to_num(nan=float("inf")).flatten().argmax().item()
                    row, col = divmod(worst, 65)
                    record = {"prime": prime_value, "k": k, "value": target_value,
                              "repeat": repeat, "bad": bad.sum().item(),
                              "max_abs": error.max().item(), "position": [row, col],
                              "actual": got[row, col].item(), "expected": want}
                    report["runs"].append(record)
                    if repeat == 0 or record["bad"]:
                        print(json.dumps(record), flush=True)
    # Valid elements and strides stay identical while guard storage changes.
    # All reads in these experiments remain inside our allocated backing tensors.
    for guard in (0.0, 1.0, 3.0, 7.0):
        a_cpu = torch.full((37 + 4096,), guard, dtype=dtype)
        b_cpu = torch.full((65 + 4096,), guard, dtype=dtype)
        a_cpu[:37] = 1
        b_cpu[:65] = 1
        a_base, b_base = a_cpu.to(args.device), b_cpu.to(args.device)
        a, b = a_base[:37].view(37, 1), b_base[:65].view(1, 65)
        c = torch.empty((37, 65), dtype=dtype, device=args.device)
        for repeat in range(args.repeats):
            launch(*inputs[32, 0.0])
            launch(a, b, c, args.variant)
            got = c.float().cpu()
            error = (got - 1).abs()
            bad = ~torch.isfinite(got) | (error > 0.05)
            index = error.nan_to_num(nan=float("inf")).flatten().argmax().item()
            row, col = divmod(index, 65)
            record = {"guard": guard, "k": 1, "repeat": repeat,
                      "bad": bad.sum().item(), "max_abs": error.max().item(),
                      "position": [row, col], "actual": got[row, col].item(),
                      "expected": 1.0}
            report["runs"].append(record)
            if repeat == 0 or record["bad"]:
                print(json.dumps(record), flush=True)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = sum(bool(x["bad"]) for x in report["runs"])
    print(f"SUMMARY {failed}/{len(report['runs'])} failed; {args.output}", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
