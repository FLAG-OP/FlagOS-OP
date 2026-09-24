#!/usr/bin/env python3
"""精度测试: 读 contiguous 黄金 → 跑候选 → 判定（值 + 连续性 + 别名）。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

IMPLS = {
    "reference": ("reference.py", "contiguous_reference"),
    "torch": ("kernel/torch_level.py", "contiguous_torch"),
    "triton": ("kernel/triton_level.py", "contiguous_triton"),
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
                    help="reference | torch | triton | <相对路径>:<入口函数>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    path, entry = args.impl.split(":") if ":" in args.impl else IMPLS[args.impl]
    fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tol = gspec["tolerance"]
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    worst = {"case": "", "dtype": "", "abs": 0.0}
    print(f"{'case':12s} {'dtype':10s} {'layout':8s} {'abs_err':>10s} 判定")
    print("-" * 52)
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        inp = d["inputs"][0].to(args.device)
        out = fn(inp)
        exp = d["expected"].float()
        abs_err = (out.float().cpu() - exp).abs().max().item()
        contig_ok = bool(out.is_contiguous())
        # 别名语义: contig 输入必须返回 self 本身
        alias_ok = (out is inp) if d["layout"] == "contig" else True
        ok = (abs_err <= tol["pointwise_abs"][d["dtype"]]
              and contig_ok and alias_ok)
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if abs_err >= worst["abs"]:
            worst.update(case=d["case"], dtype=d["dtype"], abs=abs_err)
        print(f"{d['case']:12s} {d['dtype']:10s} {d['layout']:8s} "
              f"{abs_err:>10.2e} {'✓' if ok else '✗'}"
              f"{'' if contig_ok else ' [非连续]'}"
              f"{'' if alias_ok else ' [未别名]'}")
    print("-" * 52)
    print(f"结论: {n_pass} pass / {n_fail} fail · 最差 {worst['case']}"
          f"({worst['dtype']}) abs={worst['abs']:.2e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
