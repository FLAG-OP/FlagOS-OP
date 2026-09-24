#!/usr/bin/env python3
"""精度测试: empty_like 无数值 → 只判定**元数据** (shape/dtype/stride)。

用法:
  python3 script/check_accuracy.py --impl reference --device cpu
  python3 script/check_accuracy.py --impl torch     --device mlu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

_MF = {
    "preserve_format": torch.preserve_format,
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
}

IMPLS = {
    "reference": ("reference.py", "empty_like_reference"),
    "torch": ("kernel/torch_level.py", "empty_like_torch"),
}


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="reference",
                    help="reference | torch | <相对路径>:<入口函数>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    path, entry = args.impl.split(":") if ":" in args.impl else IMPLS[args.impl]
    fn = _load(path, entry)
    yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        self_t = d["inputs"][0].to(args.device)
        out = fn(self_t, memory_format=_MF[d["memory_format"]])
        em = d["expected_meta"]
        ok = (list(out.shape) == em["shape"]
              and str(out.dtype).split(".")[-1] == em["dtype"]
              and list(out.stride()) == em["stride"])
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if not ok:
            print(f"✗ {f['file']}: got {list(out.shape)} "
                  f"{str(out.dtype).split('.')[-1]} {list(out.stride())} "
                  f"exp {em['shape']} {em['dtype']} {em['stride']}")
    print(f"结论: {n_pass} pass / {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
