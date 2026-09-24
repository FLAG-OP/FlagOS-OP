#!/usr/bin/env python3
"""精度测试（dtype 类）: 公共 dtype 判定。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
IMPLS = {"reference": ("reference.py", "result_type_reference"),
         "torch": ("kernel/torch_level.py", "result_type_torch")}


def _load(path, entry):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="reference")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()
    path, entry = args.impl.split(":") if ":" in args.impl else IMPLS[args.impl]
    fn = _load(path, entry)
    idx = json.loads((Path(args.golden) / "index.json").read_text())
    n_pass = n_fail = 0
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        pair = d["dtype_pair"]
        a = torch.empty(2, dtype=getattr(torch, pair[0]), device=args.device)
        b = torch.empty(2, dtype=getattr(torch, pair[1]), device=args.device)
        got = str(fn(a, b)).split(".")[-1]
        ok = got == d["expected"]
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if not ok:
            print(f"✗ {pair}: {got} != {d['expected']}")
    print(f"结论: {n_pass} pass / {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
