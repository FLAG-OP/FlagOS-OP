#!/usr/bin/env python3
"""FlagGems 兜底实现的黄金复验（check_accuracy.py 的 FlagGems 变体）。

check_accuracy.py 的 --impl 只有 reference/torch/triton/native；MLU 后端里
FlagGems 是 TMO→fused→math 之后的兜底，无法经 facade 单独命中。本脚本直调
`flag_gems..._cambricon.ops.attention.scaled_dot_product_attention_forward`，
容差/NaN/extreme 判定与 check_accuracy.py 完全一致，用于核对文档里
「FlagGems 精度」在扩展后的 397 组黄金上是否仍全过。

用法: python3 script/check_accuracy_flaggems.py --device mlu:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def _gems_attention():
    from flag_gems.runtime.backend._cambricon.ops.attention import (
        scaled_dot_product_attention_forward,
    )
    return scaled_dot_product_attention_forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mlu:0")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    ap.add_argument("--quiet", action="store_true", help="只打印失败与摘要")
    args = ap.parse_args()

    fn = _gems_attention()
    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tols = gspec["tolerance"]["pointwise_abs"]
    rel_tol_extreme = gspec["tolerance"].get("relative", 5e-2)
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    worst = {"case": "", "abs": 0.0}
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        ins = [t.to(args.device) for t in d["inputs"]]
        q, k, v = ins[0], ins[1], ins[2]
        m = ins[3] if len(ins) > 3 else None
        try:
            out = fn(q, k, v, m, 0.0, d["causal"], None, d["Hq"] != d["Hkv"])
            if isinstance(out, (tuple, list)):
                out = out[0]
        except Exception as e:  # 兜底实现报错即算失败
            n_fail += 1
            print(f"ERROR {f['file']}: {type(e).__name__}: {e}")
            continue
        ref = d["expected"].to(args.device)
        diff = (out.float() - ref.float()).abs()
        nan_rows = torch.isnan(ref[..., 0])
        if nan_rows.any():
            keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
            diff = diff[keep]
        err = diff.max().item() if diff.numel() else 0.0
        tol = tols[d["dtype"]]
        if d.get("case") == "extreme":
            ref_abs = ref.float().abs().clamp_min(1e-3)
            rel = ((out.float() - ref.float()).abs() / ref_abs)
            rel = torch.nan_to_num(rel, nan=0.0).max().item()
            ok = rel < rel_tol_extreme
            err = rel
        else:
            ok = err < tol
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if err > worst["abs"]:
            worst = {"case": f["file"].split("/", 1)[-1][:-3], "abs": err}
        if not ok or not args.quiet:
            name = f["file"].split("/", 1)[-1][:-3]
            print(f"{name[:40]:40s} {d['dtype']:9s} {err:10.3e} "
                  f"{'PASS' if ok else 'FAIL'}")

    print("-" * 68)
    print(f"[FlagGems] PASS {n_pass} / FAIL {n_fail}  worst={worst['abs']:.3e}"
          f" @ {worst['case']}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
