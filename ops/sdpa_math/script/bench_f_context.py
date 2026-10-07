# P800 reference probe: exact-contract math path vs no-P fused F.sdpa.
#
# This is deliberately NOT a speedup benchmark for sdpa_math:
#   * `sdpa_math_torch` and native `_scaled_dot_product_attention_math`
#     both return `(output, probabilities)` and materialize the full
#     `(B, Hq, Sq, Skv)` probability tensor.
#   * `F.scaled_dot_product_attention` returns only `output` and can select
#     a FlashAttention/efficient backend that never materializes P.
#
# Use this script only to answer the engineering question: if a downstream
# consumer does not need P, how much latency does the exact op contract leave
# on the table on the current device?
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

SHAPES = [
    ("prefill_1k_d64", 1, 16, 16, 1024, 64),
    ("prefill_1k_d128", 1, 16, 16, 1024, 128),
    ("prefill_2k_d128", 1, 16, 16, 2048, 128),
    ("gqa_1k_d128", 1, 32, 8, 1024, 128),
    ("decode_d128", 1, 16, 16, 64, 128),
    ("tail100_d64", 2, 4, 4, 100, 64),
]


def _consume(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x.reshape(-1)[0].item()


def _sync(device: str):
    import torch

    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mlu"):
        torch.mlu.synchronize()


def bench(fn, device: str, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _consume(fn())
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        _consume(fn())
    _sync(device)
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> int:
    import torch
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="float16",
                        choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    has_xmlir = False
    if args.device.startswith("cuda"):
        try:
            import torch_xmlir  # noqa: F401
            has_xmlir = True
        except ImportError:
            pass
    if has_xmlir:
        from kernel.p800_fast_level import sdpa_math_p800_fast as math_fn
    else:
        from kernel.torch_level import sdpa_math_torch as math_fn

    dtype = getattr(torch, args.dtype)
    rows = []
    header = (f"{'shape':17s} {'math(ms)':>10s} {'native(ms)':>11s} "
              f"{'F.sdpa(ms)':>11s} {'math/F':>8s}")
    print(header)
    print("-" * len(header))

    for name, batch, heads, kv_heads, seq, dim in SHAPES:
        seed = 1000 + batch + heads + seq + dim
        generator = torch.Generator(device="cpu").manual_seed(seed)

        query = (torch.randn((batch, heads, seq, dim), generator=generator)
                 .mul_(0.2).to(dtype).to(args.device))
        key = (torch.randn((batch, kv_heads, seq, dim), generator=generator)
               .mul_(0.2).to(dtype).to(args.device))
        value = (torch.randn((batch, kv_heads, seq, dim), generator=generator)
                 .mul_(0.2).to(dtype).to(args.device))
        scale = 1.0 / math.sqrt(dim)
        enable_gqa = heads != kv_heads

        math_ms = bench(
            lambda: math_fn(
                query, key, value, None, 0.0, True, None,
                scale=scale, enable_gqa=enable_gqa,
            ),
            args.device, args.warmup, args.iters,
        )
        native_ms = bench(
            lambda: torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, None, 0.0, True, None,
                scale=scale, enable_gqa=enable_gqa,
            ),
            args.device, args.warmup, args.iters,
        )
        f_ms = bench(
            lambda: F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=True, scale=scale, enable_gqa=enable_gqa,
            ),
            args.device, args.warmup, args.iters,
        )

        row = {
            "shape": name,
            "B": batch,
            "Hq": heads,
            "Hkv": kv_heads,
            "S": seq,
            "D": dim,
            "dtype": args.dtype,
            "contract": {
                "math_and_native": "returns output and P",
                "f_sdpa": "returns output only",
            },
            "math_direct_ms": round(math_ms, 4),
            "math_native_ms": round(native_ms, 4),
            "f_sdpa_ms": round(f_ms, 4),
            "math_over_f": round(math_ms / f_ms, 3),
            "native_over_f": round(native_ms / f_ms, 3),
        }
        rows.append(row)
        print(f"{name:17s} {math_ms:10.4f} {native_ms:11.4f} "
              f"{f_ms:11.4f} {math_ms / f_ms:7.2f}x")

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nsaved -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
