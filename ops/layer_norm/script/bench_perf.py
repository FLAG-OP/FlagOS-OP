#!/usr/bin/env python3
"""layer_norm 性能测试: 短采样 + 同步（issue #55 交付要求 4）。

口径: warmup 20 + iters<=100（#10）· synchronize · native/reference 对照。
用法: python3 script/bench_perf.py --device cuda:6 --json-out reports/perf_layer_norm_thead-ppu.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
CASES = [{"tokens": 4096, "hidden": 4096}, {"tokens": 8192, "hidden": 4096},
         {"tokens": 4096, "hidden": 5120}]
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
    return (time.perf_counter() - t0) / iters * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    ppu_mod = _load("ppu_mod", "kernel/thead_ppu.py")
    ref_mod = _load("ref_mod", "reference.py")

    rows = []
    for case in CASES:
        for dt in DTYPES:
            x = torch.randn(case["tokens"], case["hidden"], dtype=dt, device=dev)
            w = torch.randn(case["hidden"], dtype=dt, device=dev) * 0.1
            b = torch.randn(case["hidden"], dtype=dt, device=dev) * 0.1
            nshape = (case["hidden"],)
            native = torch.nn.functional.layer_norm(x, nshape, w, b, 1e-5)
            impls = {
                "native_impl": lambda: ppu_mod.layer_norm_ppu(x, nshape, w, b, 1e-5),
                "native_direct": lambda: torch.nn.functional.layer_norm(
                    x, nshape, w, b, 1e-5),
                "reference": lambda: ref_mod.layer_norm_reference(x, nshape, w, b, 1e-5),
            }
            nbytes = x.numel() * x.element_size() * 2
            row = {"case": str(case), "dtype": str(dt).replace("torch.", "")}
            for name, fn in impls.items():
                try:
                    t_ms = _bench(fn)
                    row[f"{name}_ms"] = round(t_ms, 4)
                    row[f"{name}_GBps"] = round(nbytes / t_ms / 1e6, 2)
                except Exception as e:
                    row[f"{name}_ms"] = None
                    row[f"{name}_error"] = repr(e)[:120]
            rows.append(row)
            print(row)

    result = {"op": "layer_norm", "device": str(dev),
              "device_name": torch.cuda.get_device_name(dev),
              "method": "warmup20+iters100+synchronize", "rows": rows}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print("saved ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
