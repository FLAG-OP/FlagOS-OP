# 性能对照实验: 本实现 vs Python F.sdpa 原生路径 vs flag_gems 直调。
# 三方同环境 A/B 对照——"修复与性能都要证据"。
#
# 运行: python3 script/bench_perf.py --device npu:0|cuda:1
#
# 方法论（对齐 FlagOS-OP 约定 + 本机实测教训）:
#   - 短采样 ≤100 次（分配器陷阱，known-issues #10）
#   - 每次读取一个输出元素到 CPU（XMLIR/CUDA 异步早退防护）
#   - 每个 dtype×shape 一组，warmup 充分（设备码编译一次，缓存后公平）
#   - flag_gems 直调不经 enable()（enable 后 SDPA 不被接管，见探测报告）
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

# bench 形状: LLaMA 类推理代表性 shape（causal，prefill/decode）
SHAPES = [
    # (tag, B, Hq, Hkv, S, D)
    ("prefill_1k_d64",   1, 16, 16, 1024, 64),
    ("prefill_1k_d128",  1, 16, 16, 1024, 128),
    ("prefill_2k_d128",  1, 16, 16, 2048, 128),
    ("prefill_4k_d128",  1, 16, 16,  4096, 128),
    ("gqa_1k_d128",      1, 32,  8, 1024, 128),
    ("decode_d128",      1, 16, 16,   64, 128),
]


def _consume_scalar(output):
    """Read one output element to force lazy/asynchronous XPU work to finish."""
    import torch
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output[0, 0, 0, 0].item()


def bench_fn(fn, dev, warmup=20, iters=100):
    import torch
    for _ in range(warmup):
        _consume_scalar(fn())
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mlu"):
        torch.mlu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _consume_scalar(fn())
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mlu"):
        torch.mlu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main() -> int:
    import torch

    from kernel.triton_level import sdpa_triton

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    dev = args.device

    def mk(tag, B, Hq, Hkv, S, D):
        g = torch.Generator(device="cpu").manual_seed(hash(tag) % 2**31)
        q = (torch.randn(B, Hq, S, D, generator=g)
             ).to(dt).to(dev)
        k = (torch.randn(B, Hkv, S, D, generator=g)
             ).to(dt).to(dev)
        v = (torch.randn(B, Hkv, S, D, generator=g)
             ).to(dt).to(dev)
        return q, k, v

    rows = []
    print(f"{'shape':18s} {'ours(ms)':>9s} {'native(ms)':>10s} "
          f"{'gems(ms)':>9s} {'speedup':>11s}")
    print("-" * 66)
    for tag, B, Hq, Hkv, S, D in SHAPES:
        q, k, v = mk(tag, B, Hq, Hkv, S, D)
        causal = True

        t_ours = bench_fn(
            lambda: sdpa_triton(q, k, v, None, 0.0, causal, None,
                                Hq != Hkv),
            dev, args.warmup, args.iters,
        )
        t_native = bench_fn(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=Hq != Hkv),
            dev, args.warmup, args.iters,
        )
        try:
            from flag_gems.runtime.backend._cambricon.ops.attention import (
                scaled_dot_product_attention_forward)
            t_gems = bench_fn(
                lambda: scaled_dot_product_attention_forward(
                    q, k, v, None, 0.0, causal, None, Hq != Hkv),
                dev, args.warmup, args.iters,
            )
        except Exception:
            try:
                from flag_gems.ops.attention import (
                    scaled_dot_product_attention_forward)
                t_gems = bench_fn(
                    lambda: scaled_dot_product_attention_forward(
                        q, k, v, None, 0.0, causal, None, Hq != Hkv),
                    dev, args.warmup, args.iters,
                )
            except Exception as e:
                t_gems = float("nan")
                print(f"    [gems fail @ {tag}] {type(e).__name__}: "
                      f"{str(e)[:80]}")

        # Keep the historical ``ours/native`` latency ratio for Ascend data,
        # but report the conventional speedup (baseline/ours) in print/JSON.
        # A number >1 therefore means "ours is faster" on every platform.
        ratio = t_ours / t_native
        speedup = t_native / t_ours
        rows.append({"shape": tag, "B": B, "Hq": Hq, "Hkv": Hkv, "S": S,
                     "D": D, "dtype": args.dtype,
                     "ours_ms": round(t_ours, 4),
                     "native_ms": round(t_native, 4),
                     "gems_ms": round(t_gems, 4) if t_gems == t_gems else None,
                     "ours_vs_native": round(ratio, 3),
                     "speedup_vs_native": round(speedup, 3)})
        print(f"{tag:18s} {t_ours:9.3f} {t_native:10.3f} "
              f"{t_gems:9.3f} {speedup:10.2f}x")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nsaved -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
