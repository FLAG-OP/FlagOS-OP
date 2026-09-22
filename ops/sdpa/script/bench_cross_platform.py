#!/usr/bin/env python3
"""跨平台 SDPA 统一基准（FA2 论文协议）—— 910 / A100 / 任意 torch 后端。

用法:
  python3 script/bench_cross_platform.py                  # 自动探测设备
  python3 script/bench_cross_platform.py --device cuda:0  # 指定
  python3 script/bench_cross_platform.py --list           # 只列配置

协议（对齐 arXiv:2307.08691 官方 benchmark）:
  batch = 16k/S（总 token 恒定），D∈{64,128}（16/32 头，hidden 2048），
  S∈{512,1k,2k,4k,8k}，causal，fp16 前向。
  实现三路: torch 原生 F.sdpa（各平台最优后端自动生效）/
           flash-attn 包（若已安装——A100 上的论文同款）/
           本库实现（ascend910 Triton / p800-kunlunxin 厂商委托，
           未支持平台自动跳过）

输出: reports/cross_platform_<device>.json（与 910 实测同 schema，
      可直接对比 / 画同图）

A100 迁移指引（目标机执行）:
  pip install torch flash-attn          # FA2 论文同款实现
  python3 script/bench_cross_platform.py --device cuda:0
  # 产物 JSON 拷回后:
  python3 script/make_a100_figs.py      # 复用同款图（数据源换新 JSON）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

TOTAL_TOKENS = 16384
S_LIST = [512, 1024, 2048, 4096, 8192]
D_LIST = [64, 128]
H_LIST = {64: 32, 128: 16}


def detect_device() -> str:
    import torch
    if torch.npu.is_available() if hasattr(torch, "npu") else False:
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def bench(fn, dev, warmup=10, iters=30):
    import torch
    for _ in range(warmup):
        fn()
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def flops(S, D, H, B, causal=True):
    return 0.5 * 4 * B * H * S * S * D if causal else 4 * B * H * S * S * D


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="如 npu:0 / cuda:0（缺省探测）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    dev = args.device or detect_device()
    if args.list:
        for D in D_LIST:
            for S in S_LIST:
                print(f"D={D} S={S} B={TOTAL_TOKENS//S} H={H_LIST[D]}")
        return 0

    # 实现可用性探测
    impls = {"native": lambda q, k, v: torch.nn.functional.
             scaled_dot_product_attention(q, k, v, is_causal=True)}

    if importlib.util.find_spec("flash_attn") is not None:
        from flash_attn import flash_attn_func
        # flash_attn 布局: (B, S, H, D)
        def _fa(q, k, v):
            return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2),
                                   v.transpose(1, 2), causal=True)
        impls["flash_attn"] = _fa

    try:
        from kernel.triton_level import sdpa_triton
        from kernel.triton_level import PLATFORM, SUPPORTED_DEVICE_TYPES
        if dev.split(":")[0] in SUPPORTED_DEVICE_TYPES:
            impls["ours_triton"] = sdpa_triton
        else:
            print(f"[skip] ours_triton: PLATFORM={PLATFORM} 绑定, "
                  f"当前 {dev}（移植见 MERGE.md）")
    except ImportError:
        pass

    dt = torch.float16
    g = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    print(f"设备: {dev} | 实现: {list(impls)}")
    print(f"{'cfg':>14} " + " ".join(f"{k:>12}" for k in impls))
    print("-" * (16 + 13 * len(impls)))
    for D in D_LIST:
        H = H_LIST[D]
        for S in S_LIST:
            B = TOTAL_TOKENS // S
            q = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
            k = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
            v = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
            fl = flops(S, D, H, B)
            row = {"S": S, "D": D, "H": H, "B": B,
                   "flops_g": round(fl / 1e9, 1)}
            line = f"D={D:<3d} S={S:<5d}"
            for name, fn in impls.items():
                t = bench(lambda: fn(q, k, v), dev, iters=args.iters)
                row[f"{name}_ms"] = round(t, 4)
                row[f"{name}_tflops"] = round(fl / (t / 1e3) / 1e12, 1)
                line += f" {row[f'{name}_tflops']:>10.1f}TF"
            rows.append(row)
            print(line, flush=True)

    slug = dev.replace(":", "_").replace("/", "_")
    out = OP_DIR / "reports" / f"cross_platform_{slug}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"device": dev, "rows": rows}, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
