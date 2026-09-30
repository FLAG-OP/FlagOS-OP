# 性能对照实验: 自研 Triton vs 原生 math 后端（同为"返回 out+概率图 P"）。
#
# 运行: python3 script/bench_perf.py --device npu:0 [--register]
#
# 口径说明（本算子的特殊性）:
#   · native baseline = 直调 `torch.ops.aten._scaled_dot_product_attention_math`
#     （未注册进程里即原生 composite，同样物化 (B,Hq,Sq,Skv) 的 P）
#     —— 与 ours 完全同算法同输出，唯一公平的 A/B。
#   · F.sdpa(MATH) 在 npu 上被 torch_npu 路由到融合注意力（不返回 P、
#     算法不同），只作参考列，不参与 speedup 判定；CPU 上 F.sdpa(MATH)
#     经注册会打到本算子，故不在本脚本里测。
#   · --register 额外测 A1 注册后的 torch.ops 调用（含 autograd.Function
#     包装开销），验证"接管不带来额外税"。
# 方法论继承 ops/sdpa: 短采样、读一个输出元素防异步早退、warmup 充分。
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

# 形状: prefill / decode / GQA（causal 为主）
SHAPES = [
    ("prefill_1k_d64",   1, 16, 16, 1024,  64),
    ("prefill_1k_d128",  1, 16, 16, 1024, 128),
    ("prefill_2k_d128",  1, 16, 16, 2048, 128),
    ("gqa_1k_d128",      1, 32,  8, 1024, 128),
    ("decode_d128",      1, 16, 16,   64, 128),
    ("tail100_d64",      2,  4,  4,  100,  64),
]


def _consume(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output[0, 0, 0, 0].item()


def _sync(dev):
    import torch
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mlu"):
        torch.mlu.synchronize()


def bench_fn(fn, dev, warmup=20, iters=50):
    for _ in range(warmup):
        _consume(fn())
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        _consume(fn())
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main() -> int:
    import torch

    from register import register_a1

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--register", action="store_true",
                    help="同时测 A1 注册后的 torch.ops 路径")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    dev = args.device
    # ours 按平台选: triton 绑定 ascend910，CPU 上用 ATen 组合（同语义）
    if dev.startswith("cpu"):
        from kernel.torch_level import sdpa_math_torch as ours_fn
    else:
        from kernel.triton_level import sdpa_math_triton as ours_fn
    # 注册推迟到 native 基线测完（注册后 torch.ops 就是本实现，测不到原生）
    libs = None

    def mk(tag, B, Hq, Hkv, S, D):
        g = torch.Generator(device="cpu").manual_seed(hash(tag) % 2 ** 31)
        q = torch.randn(B, Hq, S, D, generator=g).to(dt).to(dev)
        k = torch.randn(B, Hkv, S, D, generator=g).to(dt).to(dev)
        v = torch.randn(B, Hkv, S, D, generator=g).to(dt).to(dev)
        return q, k, v

    hdr = f"{'shape':17s} {'ours(ms)':>9s} {'native(ms)':>10s} " \
          f"{'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for tag, B, Hq, Hkv, S, D in SHAPES:
        q, k, v = mk(tag, B, Hq, Hkv, S, D)
        causal = True
        gqa = Hq != Hkv

        t_ours = bench_fn(
            lambda: ours_fn(q, k, v, None, 0.0, causal, None,
                            enable_gqa=gqa),
            dev, args.warmup, args.iters)
        t_native = bench_fn(
            lambda: torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, None, 0.0, causal, None, enable_gqa=gqa),
            dev, args.warmup, args.iters)

        speedup = t_native / t_ours
        row = {"shape": tag, "B": B, "Hq": Hq, "Hkv": Hkv, "S": S, "D": D,
               "dtype": args.dtype, "ours_ms": round(t_ours, 4),
               "native_ms": round(t_native, 4),
               "speedup_vs_native": round(speedup, 3)}
        print(f"{tag:17s} {t_ours:9.3f} {t_native:10.3f} {speedup:7.2f}x")
        rows.append(row)

    if args.register:
        # 原生基线已采完 → 现在接管，再测 A1 路径（含 autograd.Function 包装）
        libs = register_a1("AutogradPrivateUse1" if dev.startswith("npu")
                           else "CPU", None,
                           "triton" if dev.startswith("npu") else "torch")
        for row in rows:
            tag, B, Hq, Hkv, S, D = (row["shape"], row["B"], row["Hq"],
                                     row["Hkv"], row["S"], row["D"])
            q, k, v = mk(tag, B, Hq, Hkv, S, D)
            t_a1 = bench_fn(
                lambda: torch.ops.aten._scaled_dot_product_attention_math(
                    q, k, v, None, 0.0, True, None, enable_gqa=Hq != Hkv),
                dev, args.warmup, args.iters)
            row["a1_ms"] = round(t_a1, 4)
            row["a1_vs_native"] = round(row["native_ms"] / t_a1, 3)
            print(f"  [A1] {tag:17s} {t_a1:8.3f}ms "
                  f"(vs native {row['native_ms']:.3f}ms, "
                  f"{row['native_ms'] / t_a1:.2f}x)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nsaved -> {args.json_out}")
    if libs is not None:
        print("note: torch.library 注册不可撤销（同进程保留到退出）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
