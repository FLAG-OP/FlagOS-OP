#!/usr/bin/env python3
"""spmm 性能测试: 短采样 + 同步（issue #57 交付与验收）。

口径: warmup 20 + iters<=100（#10）· synchronize · 三方对照
（index_add 组合 / 原生[预期缺证: cusparse 缺口, 见 probe 报告] / fp64 参考）。
用法: python3 script/bench_perf.py --device cuda:6 --json-out reports/perf_spmm_thead-ppu.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
CASES = [  # (M, K, N, density) — Gowalla 型 + 台账 N=64/512 缩样
    (8192, 64, 64, 0.01),
    (16384, 64, 512, 0.005),
    (4096, 128, 64, 0.02),
]


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
    for m, k, n, dens in CASES:
        dense = torch.rand(m, k, device=dev) * (torch.rand(m, k, device=dev) < dens)
        a = dense.to_sparse_csr()
        b = torch.randn(k, n, device=dev)
        nnz = int(a.values().shape[0])
        flops = 2 * nnz * n
        impls = {
            "index_add": lambda: ppu_mod.spmm_ppu(a, b),
            "native": lambda: torch.sparse.mm(a, b),
            "reference_fp64": lambda: ref_mod.spmm_reference(a, b),
        }
        row = {"M": m, "K": k, "N": n, "density": dens, "nnz": nnz}
        for name, fn in impls.items():
            try:
                t_ms = _bench(fn)
                row[f"{name}_ms"] = round(t_ms, 4)
                row[f"{name}_GFLOPS"] = round(flops / t_ms / 1e6, 2)
            except Exception as e:
                row[f"{name}_ms"] = None
                row[f"{name}_error"] = repr(e)[:120]
        rows.append(row)
        print(row)

    result = {"op": "spmm", "device": str(dev),
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
