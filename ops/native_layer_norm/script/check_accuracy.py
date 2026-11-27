#!/usr/bin/env python3
"""native_layer_norm 黄金精度校验（issue #55 要求 3）。

用法:
  python3 script/check_accuracy.py --impl native --device cuda:6   # 真机
  python3 script/check_accuracy.py --impl torch   --device cuda:6  # torch 级
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

    if args.impl == "native":
        fwd = _load("impl_mod", "kernel/thead_ppu.py").native_layer_norm_ppu
    else:
        fwd = _load("torch_mod", "kernel/torch_level.py").native_layer_norm_torch

    index = json.loads((HERE / "goldendata/data/index.json").read_text())
    dev = torch.device(args.device) if args.device else torch.device("cpu")
    if args.device and dev.type == "cuda":
        torch.cuda.set_device(dev)

    passed, failed, worst = 0, 0, 0.0
    for entry in index["files"]:
        payload = torch.load(HERE / "goldendata/data" / entry["file"],
                             weights_only=False)
        x = payload["input"].to(dev)
        w = payload["weight"].to(dev)
        b = payload["bias"].to(dev)
        nshape = payload["normalized_shape"]
        eps = payload["eps"]
        expected = payload["expected"].to(dev)
        got = fwd(x, nshape, w, b, eps)
        if isinstance(got, (tuple, list)):
            got = got[0]
        tol = 2e-5 if entry["dtype"] == "float32" else 1e-2
        metric = (got.float() - expected.float()).abs().max().item()
        ok = metric < tol
        passed, failed = passed + ok, failed + (not ok)
        if ok:
            worst = max(worst, metric)
        else:
            print("FAIL", entry["file"], "err=", metric)


    total = passed + failed
    print(f"native_layer_norm:", args.impl, "@", dev, passed, "/", total, "PASS, worst=%.3e" % worst)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
