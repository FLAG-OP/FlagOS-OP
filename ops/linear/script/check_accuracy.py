#!/usr/bin/env python3
"""linear 黄金精度校验（GEMM 5e-2 rel 判卷, issue #55 要求 3）。

用法:
  python3 script/check_accuracy.py --impl native --device cuda:6
  python3 script/check_accuracy.py --impl torch   --device cuda:6
  python3 script/check_accuracy.py --impl native                   # CPU 自检
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", choices=["native", "torch"], default="native")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ref_mod = _load("linear_reference", "reference.py")
    if args.impl == "native":
        impl_mod = _load("linear_ppu", "kernel/thead_ppu.py")
        fwd = impl_mod.linear_ppu
    else:
        impl_mod = _load("linear_torch", "kernel/torch_level.py")
        fwd = impl_mod.linear_torch

    index = json.loads((HERE / "goldendata/data/index.json").read_text())
    dev = torch.device(args.device) if args.device else torch.device("cpu")
    if args.device and dev.type == "cuda":
        torch.cuda.set_device(dev)

    passed, failed, worst = 0, 0, 0.0
    for entry in index["files"]:
        payload = torch.load(HERE / "goldendata/data" / entry["file"],
                             weights_only=False)
        x = payload["x"].to(dev)
        w = payload["w"].to(dev)
        b = payload["bias"]
        b = b.to(dev) if b is not None else None
        expected = payload["expected"].to(dev)
        got = fwd(x, w, b)
        rel = ((got.float() - expected.float()).abs()
               / expected.float().abs().clamp_min(1.0)).max().item()
        ok = rel < 5e-2
        passed, failed = passed + ok, failed + (not ok)
        worst = max(worst, rel) if ok else worst
        if not ok:
            print(f"FAIL {entry['file']} rel={rel}")
    total = passed + failed
    print(f"[linear:{args.impl}@{dev}] {passed}/{total} PASS, worst_rel={worst:.3e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
