#!/usr/bin/env python3
"""linear 性能测试: 短采样 + 同步（issue #55 交付要求 4）。

用法:
  python3 script/bench_perf.py --device cuda:6 \
      --json-out reports/perf_fp16_thead-ppu.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
SHAPES = [(4096, 4096), (8192, 8192), (4096, 11008), (1, 4096)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ppu_mod = _load("linear_ppu", "kernel/thead_ppu.py")
    torch_mod = _load("linear_torch", "kernel/torch_level.py")
    ref_mod = _load("linear_reference", "reference.py")
    impls = {
        "native": lambda x, w: ppu_mod.linear_ppu(x, w, None),
        "torch": lambda x, w: torch_mod.linear_torch(x, w, None),
        "reference": lambda x, w: ref_mod.linear_reference(x, w, None),
    }

    rows = []
    for m, k in SHAPES:
        for n_out in (k,):
            for dt in DTYPES:
                x = torch.randn(m, k, dtype=torch.float32, device=dev).to(dt)
                w = (torch.randn(n_out, k, dtype=torch.float32,
                                 device=dev) * 0.02).to(dt)
                flops = 2 * m * k * n_out
                row = {"M": m, "K": k, "N": n_out,
                       "dtype": str(dt).replace("torch.", "")}
                for name, fn in impls.items():
                    try:
                        t_ms = _bench(lambda: fn(x, w))
                        row[f"{name}_ms"] = round(t_ms, 4)
                        row[f"{name}_TFLOPS"] = round(flops / t_ms / 1e9, 2)
                    except Exception as e:
                        row[f"{name}_ms"] = None
                        row[f"{name}_error"] = repr(e)[:120]
                if row.get("native_ms") and row.get("torch_ms"):
                    row["torch_vs_native"] = round(
                        row["torch_ms"] / row["native_ms"], 3)
                rows.append(row)
                print(row)

    result = {"op": "linear", "device": str(dev),
              "device_name": torch.cuda.get_device_name(dev),
              "method": "warmup20+iters100+synchronize, no-bias",
              "rows": rows}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
