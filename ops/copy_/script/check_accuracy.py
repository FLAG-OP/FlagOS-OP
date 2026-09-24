#!/usr/bin/env python3
"""精度测试: 读 copy_ 黄金 → 跑候选（原地）→ 判定值/返回/原地性。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

IMPLS = {
    "reference": ("reference.py", "copy_reference"),
    "torch": ("kernel/torch_level.py", "copy_torch"),
    "triton": ("kernel/triton_level.py", "copy_triton"),
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
    print(f"{'case':10s} {'dst dtype':10s} {'src dtype':10s} {'abs_err':>10s} 判定")
    print("-" * 56)
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        dst = d["inputs"][0].to(args.device)      # 每次新建，保证原地干净
        src = d["inputs"][1].to(args.device)
        out = fn(dst, src)
        exp = d["expected"]
        outf, expf = out.float().cpu(), exp.float()
        # 处理 boundary 溢出到 ±inf / nan 的合法 cast：isclose(inf,inf)=True
        tol_v = tol["pointwise_abs"][d["dtype"]]
        close = torch.isclose(outf, expf, rtol=0.0, atol=tol_v,
                              equal_nan=True)
        match = bool(close.all().item())
        finite = torch.isfinite(outf) & torch.isfinite(expf)
        abs_err = ((outf[finite] - expf[finite]).abs().max().item()
                   if finite.any() else 0.0)
        inplace_ok = out is dst
        dtype_ok = out.dtype == d["inputs"][0].dtype
        ok = match and inplace_ok and dtype_ok
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if abs_err >= worst["abs"]:
            worst.update(case=d["case"], dtype=d["dtype"], abs=abs_err)
        print(f"{d['case']:10s} {d['dtype']:10s} {d['src_dtype']:10s} "
              f"{abs_err:>10.2e} {'✓' if ok else '✗'}"
              f"{'' if inplace_ok else ' [非原地]'}")
    print("-" * 56)
    print(f"结论: {n_pass} pass / {n_fail} fail · 最差 {worst['case']}"
          f"({worst['dtype']}) abs={worst['abs']:.2e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
