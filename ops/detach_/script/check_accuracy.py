#!/usr/bin/env python3
"""精度测试（alias 类）: 元数据 + 共享存储 + requires_grad=False。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
IMPLS = {"reference": ("reference.py", "detach__reference"),
         "torch": ("kernel/torch_level.py", "detach__torch")}


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
        x = d["inputs"][0].to(args.device)
        x.requires_grad_(True)
        out = fn(x)
        em = d["expected_meta"]
        ok = (list(out.shape) == em["shape"]
              and str(out.dtype).split(".")[-1] == em["dtype"]
              and list(out.stride()) == em["stride"]
              and bool(out.requires_grad) == em["requires_grad"]
              and out.data_ptr() == x.data_ptr())
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if not ok:
            print(f"✗ {f['file']}")
    print(f"结论: {n_pass} pass / {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
