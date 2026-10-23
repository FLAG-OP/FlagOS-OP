#!/usr/bin/env python3
"""Compile P800 matmul dot variants without launching a device kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR.parent.parent))


@triton.jit
def _dot_once(a, b, out, block: tl.constexpr):
    row = tl.arange(0, block)
    col = tl.arange(0, block)
    left = tl.load(a + row[:, None] * block + col[None, :])
    right = tl.load(b + row[:, None] * block + col[None, :])
    result = tl.dot(left, right, allow_tf32=False)
    tl.store(out + row[:, None] * block + col[None, :], result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("minimal", "full"), required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = ap.parse_args()

    from flag_gems.runtime import torch_device_fn
    from ops.matmul.kernel.triton_level import _matmul_kernel

    block = 32
    a = torch.empty((1, block, block), device=args.device,
                    dtype=getattr(torch, args.dtype))
    b = torch.empty_like(a)
    out = torch.empty_like(a)
    with torch_device_fn.device(a.device):
        if args.kind == "minimal":
            _dot_once.warmup(a, b, out, block=block, grid=(1,), num_warps=4)
        else:
            _matmul_kernel.warmup(
                a, b, out, block, block, block,
                *a.stride(), *b.stride(), *out.stride(),
                BM=block, BN=block, BK=block, grid=(1, 1), num_warps=4,
            )
    print(f"dot compile passed: {args.kind}, {args.dtype}, {args.device}")


if __name__ == "__main__":
    main()
