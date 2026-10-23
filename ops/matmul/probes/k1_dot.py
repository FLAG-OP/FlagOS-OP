#!/usr/bin/env python3
"""Compare the current P800 K=1 dot path with diagnostic outer products."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ops.matmul.kernel.triton_level import _matmul_kernel
from ops.matmul.probes.outer import matmul_outer


def dot_direct(a: torch.Tensor, b: torch.Tensor,
               output_fill: float | None = None) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if output_fill is not None:
        out.fill_(output_fill)
    bm = bn = bk = 32
    from flag_gems.runtime import torch_device_fn

    with torch_device_fn.device(a.device):
        _matmul_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), 1)](
            a, b, out, m, n, k,
            0, *a.stride(), 0, *b.stride(), 0, *out.stride(),
            BM=bm, BN=bn, BK=bk, num_warps=4,
        )
    return out


def compare(actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual.float().cpu() - expected.float().cpu()).abs()
    scale = expected.float().abs().max().item()
    rel = diff.max().item() / max(scale, 1e-6)
    if not torch.isfinite(diff).all().item():
        rel = float("inf")
    index = diff.flatten().argmax().item()
    row, col = divmod(index, diff.shape[1])
    return rel, (row, col), actual[row, col].item(), expected[row, col].item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--dtype", choices=("all", "bfloat16", "float16", "float32"),
                    default="all")
    ap.add_argument("--output-fill", type=float)
    args = ap.parse_args()
    golden = ROOT / "ops/matmul/goldendata/data"

    for dtype in ("bfloat16", "float16", "float32"):
        if args.dtype != "all" and dtype != args.dtype:
            continue
        for seed in range(3):
            for case in ("rank_one", "matrix_vector_grad_a"):
                source = "rank_one" if case == "rank_one" else "matrix_vector"
                data = torch.load(golden / f"{source}_{dtype}_s{seed}.pt",
                                  map_location="cpu", weights_only=True)
                if case == "rank_one":
                    a, b = data["inputs"]
                else:
                    a = data["grad_output"].unsqueeze(-1)
                    b = data["inputs"][1].unsqueeze(0)
                expected = (a.float() @ b.float()).to(a.dtype)
                a, b = a.to(args.device), b.to(args.device)
                outer = matmul_outer(a, b)
                outer_rel = compare(outer, expected)[0]
                failures = 0
                worst = (0.0, (0, 0), 0.0, 0.0)
                for _ in range(args.repeats):
                    result = compare(dot_direct(a, b, args.output_fill), expected)
                    failures += result[0] > 0.05
                    if result[0] > worst[0]:
                        worst = result
                print(f"{case} {dtype} seed={seed} strides={a.stride()}/{b.stride()} "
                      f"dot_fail={failures}/{args.repeats} dot_worst={worst[0]:.4g} "
                      f"at={worst[1]} got={worst[2]:.6g} expected={worst[3]:.6g} "
                      f"outer_rel={outer_rel:.4g}", flush=True)

    data = torch.load(golden / "rank_one_bfloat16_s0.pt", map_location="cpu",
                      weights_only=True)
    left, right = (x.to(args.device) for x in data["inputs"])
    expected = (left.float().cpu() @ right.float().cpu()).to(left.dtype)
    for k in (1, 2, 4, 8, 16, 32, 33, 64):
        a = torch.zeros((left.shape[0], k), device=args.device, dtype=left.dtype)
        b = torch.zeros((k, right.shape[1]), device=args.device, dtype=right.dtype)
        a[:, 0] = left[:, 0]
        b[0, :] = right[0, :]
        result = dot_direct(a, b, args.output_fill)
        rel, index, got, want = compare(result, expected)
        print(f"zero_padded_k={k} rel={rel:.4g} at={index} "
              f"got={got:.6g} expected={want:.6g}", flush=True)

    for m, n in ((1, 1), (2, 2), (32, 32), (32, 33), (33, 32), (37, 65)):
        a, b = left[:m], right[:, :n]
        expected = (a.float().cpu() @ b.float().cpu()).to(a.dtype)
        rel, index, got, want = compare(dot_direct(a, b, args.output_fill), expected)
        print(f"tile_shape=({m},{n}) rel={rel:.4g} at={index} "
              f"got={got:.6g} expected={want:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
