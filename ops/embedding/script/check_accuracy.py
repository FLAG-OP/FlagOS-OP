#!/usr/bin/env python3
"""Check a candidate embedding implementation against CPU goldens."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

IMPLS = {
    "reference": ("reference.py", "embedding_reference"),
    "torch": ("kernel/torch_level.py", "embedding_torch"),
    "p800": ("kernel/p800_kunlunxin.py", "embedding"),
    "cambricon": ("kernel/cambricon.py", "embedding"),
    "triton": ("kernel/triton_level.py", "embedding_triton"),
    "native": ("kernel/_native_shim.py", "embedding_native"),
}


def _load(path, entry):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="p800")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    fn = _load(*IMPLS[args.impl])
    spec = yaml.safe_load(
        (HERE / "goldendata/inputs_spec.yaml").read_text()
    )
    tolerance = spec["tolerance"]["pointwise_abs"]
    index = json.loads((Path(args.golden) / "index.json").read_text())
    n_pass = n_fail = 0
    worst = 0.0
    for item in index["files"]:
        data = torch.load(Path(args.golden) / item["file"])
        weight, indices = (x.to(args.device) for x in data["inputs"])
        out = fn(
            weight, indices, data["padding_idx"], False, False
        )
        diff = (out.float().cpu() - data["expected"].float()).abs()
        err = diff.max().item() if diff.numel() else 0.0
        ok = err <= tolerance[data["dtype"]]
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        worst = max(worst, err)
        print(f"{item['file']:<44} {err:8.3e} "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 60)
    print(f"PASS {n_pass} / FAIL {n_fail}; worst={worst:.3e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
