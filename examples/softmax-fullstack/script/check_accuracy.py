#!/usr/bin/env python3
"""读黄金数据 → 跑 softmax 候选实现 → 按容差判定。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

IMPLS = {
    "reference": ("reference.py", "softmax_reference"),
    "torch": ("kernel/torch_level.py", "softmax_torch"),
    "triton": ("kernel/triton_level.py", "softmax_triton"),
}


def _load(path, entry):
    spec = importlib.util.spec_from_file_location(
        "acc_" + path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="triton",
                    help="reference | torch | triton | <相对路径>:<入口>")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    path, entry = (args.impl.split(":") if ":" in args.impl
                   else IMPLS[args.impl])
    fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tol = gspec["tolerance"]
    idx = json.loads((HERE / "goldendata/index.json").read_text())

    n_pass = n_fail = 0
    print(f"{'case':10s} {'dtype':10s} {'abs_err':>10s} 判定")
    print("-" * 42)
    for f in idx["files"]:
        d = torch.load(HERE / "goldendata" / f["file"])
        out = fn(d["inputs"][0].to(args.device)).float().cpu()
        exp = d["expected"].float()
        err = (out - exp).abs().max().item()
        ok = err <= tol["pointwise_abs"][d["dtype"]]
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        print(f"{d['case']:10s} {d['dtype']:10s} {err:>10.2e} "
              f"{'✓' if ok else '✗'}")
    print("-" * 42)
    print(f"结论: {n_pass} pass / {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
