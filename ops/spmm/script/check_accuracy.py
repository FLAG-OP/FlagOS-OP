#!/usr/bin/env python3
"""spmm 黄金精度校验（issue #57 交付与验收）。

用法:
  python3 script/check_accuracy.py --impl native --device cuda:6   # index_add 组合
  python3 script/check_accuracy.py --impl torch                    # CPU 官方对拍
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

    fwd = _load("impl_mod", "kernel/thead_ppu.py").spmm_ppu

    index = json.loads((HERE / "goldendata/data/index.json").read_text())
    dev = torch.device(args.device) if args.device else torch.device("cpu")
    if args.device and dev.type == "cuda":
        torch.cuda.set_device(dev)

    passed, failed, worst = 0, 0, 0.0
    for entry in index["files"]:
        payload = torch.load(HERE / "goldendata/data" / entry["file"],
                             weights_only=False)
        csr = payload["csr"].to(dev)
        b = payload["dense"].to(dev)
        expected = payload["expected"].to(dev)
        got = fwd(csr, b)
        # 乘加放大形态 → GEMM 档 5e-2 rel（acceptance.md; fp64 参考判卷）
        metric = ((got.float() - expected.float()).abs()
                  / expected.float().abs().clamp_min(1.0)).max().item()
        ok = metric < 5e-2
        passed, failed = passed + ok, failed + (not ok)
        if ok:
            worst = max(worst, metric)
        else:
            print("FAIL", entry["file"], "rel=", metric)

    total = passed + failed
    print("spmm:", args.impl, "@", dev, passed, "/", total,
          "PASS, worst=%.3e" % worst)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
