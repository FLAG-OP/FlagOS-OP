#!/usr/bin/env python3
"""Benchmark the FLAG-OP/gatherFIX stride-aware gather algorithm on P800.

The organization fix targets Ascend ``torch.gather`` correctness for
non-contiguous indices.  This probe specializes its rank-5 stride-aware kernel
to ``aten::embedding`` by gathering an expanded ``[M, D]`` index view along
weight dim 0.  It is evidence only; the production embedding backend remains
the XMLIR native row gather.
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

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit
def _gatherfix_stride_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    idx_shape0,
    idx_shape1,
    idx_shape2,
    idx_shape3,
    idx_shape4,
    idx_stride0,
    idx_stride1,
    idx_stride2,
    idx_stride3,
    idx_stride4,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    dim_stride,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    # Algorithm adapted from FLAG-OP/gatherFIX src/gather.py (Apache-2.0).
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offset < total

    cur = offset
    coord0 = cur % idx_shape0
    cur = cur // idx_shape0
    coord1 = cur % idx_shape1
    cur = cur // idx_shape1
    coord2 = cur % idx_shape2
    cur = cur // idx_shape2
    coord3 = cur % idx_shape3
    cur = cur // idx_shape3
    coord4 = cur % idx_shape4

    index_offset = (
        coord0 * idx_stride0 + coord1 * idx_stride1
        + coord2 * idx_stride2 + coord3 * idx_stride3
        + coord4 * idx_stride4
    )
    out_offset = (
        coord0 * out_stride0 + coord1 * out_stride1
        + coord2 * out_stride2 + coord3 * out_stride3
        + coord4 * out_stride4
    )
    # For embedding dim=0, restride_dim(weight, 0, index.shape) yields logical
    # input strides (0, weight_stride1, 0, 0, 0).
    base = (
        coord0 * inp_stride0 + coord1 * inp_stride1
        + coord2 * inp_stride2 + coord3 * inp_stride3
        + coord4 * inp_stride4
    )

    row = tl.load(index_ptr + index_offset, mask=mask, other=0)
    value = tl.load(
        weight_ptr + base + row * dim_stride, mask=mask, other=0.0
    )
    tl.store(out_ptr + out_offset, value, mask=mask)


def embedding_gatherfix(weight, indices):
    """Row-lookup specialization of the organization stride-aware gather."""
    flat_indices = indices.reshape(-1).contiguous()
    m, dim = flat_indices.numel(), weight.shape[-1]
    expanded = flat_indices[:, None].expand(-1, dim)
    out = torch.empty_like(expanded, dtype=weight.dtype)
    total = out.numel()
    block = 1024
    grid = (triton.cdiv(total, block),)

    # Expanded rank-2 tensors padded to the fixed rank-5 kernel signature.
    shapes = (*expanded.shape, 1, 1, 1)
    idx_strides = (*expanded.stride(), 0, 0, 0)
    out_strides = (*out.stride(), 0, 0, 0)
    # Logical restride_dim(weight, dim=0, index.shape) -> shape (M,D),
    # strides (0, weight.stride(1)).
    inp_strides = (0, weight.stride(1), 0, 0, 0)

    with torch_device_fn.device(weight.device):
        _gatherfix_stride_kernel[grid](
            weight,
            expanded,
            out,
            *shapes,
            *idx_strides,
            *out_strides,
            *inp_strides,
            weight.stride(0),
            total,
            BLOCK_SIZE=block,
            num_warps=4,
            num_stages=1,
        )
    return out.view(*indices.shape, dim)


def _consume(output):
    return output.reshape(-1)[0].item()


def _bench(call, warmup, iters):
    for _ in range(warmup):
        _consume(call())
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _consume(call())
        samples.append((time.perf_counter() - t0) * 1000)
    torch.cuda.synchronize()
    return sorted(samples)[len(samples) // 2]


def main() -> int:
    from common.xpu_compat import ensure_xpu_compiler_debuggable
    from kernel.p800_kunlunxin import embedding as production

    ensure_xpu_compiler_debuggable()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    shapes = [
        (1024, 4096, 128),
        (16384, 4096, 128),
        (131072, 4096, 128),
        (16384, 4096, 512),
        (16384, 128256, 128),
    ]
    rows = []
    for m, num_weights, dim in shapes:
        torch.manual_seed(1000 + m + dim)
        weight = (
            torch.randn(num_weights, dim) * 0.1
        ).to(torch.float16).to(args.device)
        indices = torch.randint(
            0, num_weights, (m,), device=args.device
        )
        ref = torch.ops.aten.embedding(weight, indices, -1, False, False)
        gatherfix = embedding_gatherfix(weight, indices)
        torch.cuda.synchronize()
        err = (gatherfix.float() - ref.float()).abs().max().item()
        assert err == 0.0, (m, num_weights, dim, err)

        native_ms = _bench(
            lambda: torch.ops.aten.embedding(
                weight, indices, -1, False, False
            ), args.warmup, args.iters,
        )
        production_ms = _bench(
            lambda: production(weight, indices), args.warmup, args.iters
        )
        gatherfix_ms = _bench(
            lambda: embedding_gatherfix(weight, indices),
            args.warmup, args.iters,
        )
        row = {
            "M": m,
            "num_weights": num_weights,
            "dim": dim,
            "native_ms": round(native_ms, 4),
            "production_ms": round(production_ms, 4),
            "gatherfix_specialized_ms": round(gatherfix_ms, 4),
            "gatherfix_over_native": round(gatherfix_ms / native_ms, 3),
            "max_err": err,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "source": "https://github.com/FLAG-OP/gatherFIX",
            "device": args.device,
            "completion_guard": "reshape(-1)[0].item()",
            "rows": rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
