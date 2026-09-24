#!/usr/bin/env python3
"""精度测试: 读 type_as 黄金数据 → 跑候选实现 → 按规格容差判定。

容差按 **目标 dtype**（输出精度）取，而非输入 dtype。

用法:
  python3 script/check_accuracy.py --impl reference --device cpu
  python3 script/check_accuracy.py --impl triton --device mlu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

IMPLS = {
    "reference": ("reference.py", "type_as_reference"),
    "torch": ("kernel/torch_level.py", "type_as_torch"),
    "triton": ("kernel/triton_level.py", "type_as_triton"),
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

    if ":" in args.impl:
        path, entry = args.impl.split(":")
    else:
        path, entry = IMPLS[args.impl]
    fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tol = gspec["tolerance"]
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    worst = {"case": "", "dtype": "", "abs": 0.0, "rel": 0.0}
    print(f"{'case':12s} {'in->out':22s} {'abs_err':>10s} {'rel_err':>10s} 判定")
    print("-" * 70)
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        ins = [t.to(args.device) for t in d["inputs"]]
        out = fn(*ins).float().cpu()
        exp = d["expected"].float()
        abs_err = (out - exp).abs().max().item()
        rel_err = abs_err / (exp.abs().max().item() + 1e-6)
        limit = tol["pointwise_abs"][d["target_dtype"]]
        ok = abs_err <= limit or rel_err <= tol["relative"]
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if abs_err >= worst["abs"]:
            worst.update(case=d["case"], dtype=d["target_dtype"],
                         abs=abs_err, rel=rel_err)
        tag = f"{d['dtype']}->{d['target_dtype']}"
        print(f"{d['case']:12s} {tag:22s} {abs_err:>10.2e} "
              f"{rel_err:>10.2e} {'✓' if ok else '✗'}")
    print("-" * 70)
    print(f"结论: {n_pass} pass / {n_fail} fail · 最差 {worst['case']}"
          f"(out={worst['dtype']}) abs={worst['abs']:.2e} "
          f"rel={worst['rel']:.2e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
