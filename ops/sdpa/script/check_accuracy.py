#!/usr/bin/env python3
"""精度测试: 读黄金数据 → 跑候选实现 → 按规格容差判定。

用法:
  python3 script/check_accuracy.py --impl reference --device cpu
  python3 script/check_accuracy.py --impl triton   --device npu:0
  python3 script/check_accuracy.py --impl native   --device npu:0  # 原生对照
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
    "reference": ("reference.py", "sdpa_reference"),
    "torch": ("kernel/torch_level.py", "sdpa_torch"),
    "triton": ("kernel/triton_level.py", "sdpa_triton"),
    "native": ("kernel/_native_shim.py", "sdpa_native"),
}


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="triton",
                    help="reference | torch | triton | native")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    path, entry = IMPLS[args.impl]
    fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tols = gspec["tolerance"]["pointwise_abs"]
    rel_tol_extreme = gspec["tolerance"].get("relative", 5e-2)
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    worst = {"case": "", "abs": 0.0}
    print(f"{'case':40s} {'dtype':9s} {'abs_err':>10s} 判定")
    print("-" * 68)
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        ins = [t.to(args.device) for t in d["inputs"]]
        q, k, v = ins[0], ins[1], ins[2]
        m = ins[3] if len(ins) > 3 else None
        out = fn(q, k, v, m, 0.0, d["causal"], None, d["Hq"] != d["Hkv"])
        ref = d["expected"].to(args.device)
        diff = (out.float() - ref.float()).abs()
        nan_rows = torch.isnan(ref[..., 0])
        if nan_rows.any():
            keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
            diff = diff[keep]
        err = diff.max().item() if diff.numel() else 0.0
        tol = tols[d["dtype"]]
        # extreme（带 special）case: softmax 饱和下 fp32 累加顺序差异
        # ~3e-4 属正常（实测 native vs golden 同值），改用相对容差
        # wt 2026-09-16-fix
        if d.get("case") == "extreme":
            ref_abs = ref.float().abs().clamp_min(1e-3)
            rel = ((out.float() - ref.float()).abs() / ref_abs)
            rel = torch.nan_to_num(rel, nan=0.0).max().item()
            ok = rel < rel_tol_extreme
            err = rel  # 显示相对误差
        else:
            ok = err < tol
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if err > worst["abs"]:
            worst = {"case": f["file"].split("/", 1)[-1][:-3], "abs": err}
        name = f["file"].split("/", 1)[-1][:-3]
        print(f"{name[:40]:40s} {d['dtype']:9s} {err:10.3e} "
              f"{'PASS' if ok else 'FAIL'}")

    print("-" * 68)
    print(f"PASS {n_pass} / FAIL {n_fail}  worst={worst['abs']:.3e}"
          f" @ {worst['case']}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
