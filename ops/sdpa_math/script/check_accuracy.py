#!/usr/bin/env python3
"""精度测试: 读黄金数据 → 跑候选实现 → 按规格容差判定（out 与 P 双判卷）。

用法:
  python3 script/check_accuracy.py --impl reference --device cpu
  python3 script/check_accuracy.py --impl torch     --device cpu
  python3 script/check_accuracy.py --impl triton    --device npu:0
  python3 script/check_accuracy.py --impl native    --device cpu   # 原生对照

注意:
  · --impl native 必须在**未注册**的进程里跑（本脚本不注册 → 即原生）；
    bool attn_mask 用例会跳过（native 直调 0/1 加性怪癖，见 reference 顶注）。
  · 判卷口径是 out 与 attn_probs 两路，dropout 用例带显式 dropout_mask
    （跨设备确定性）。
"""
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
    "reference": ("reference.py", "sdpa_math_reference"),
    "torch": ("kernel/torch_level.py", "sdpa_math_torch"),
    "triton": ("kernel/triton_level.py", "sdpa_math_triton"),
}


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def _native(q, k, v, m, drop_p, causal, dm, scale, gqa):
    """未注册进程里 = 原生实现（直调本算子）。"""
    return torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, m, drop_p, causal, dm, scale=scale, enable_gqa=gqa)


def _run_case(fn, d, ins, device):
    q, k, v = (t.to(device) for t in ins[:3])
    m = ins[3].to(device) if len(ins) > 3 else None
    dm = d.get("dropout_mask")
    dm = dm.to(device) if dm is not None else None
    scale = d.get("scale_param")
    if fn is _native:
        return _native(q, k, v, m, d["dropout_p"], d["causal"], dm, scale,
                       d["enable_gqa"])
    return fn(q, k, v, m, d["dropout_p"], d["causal"], dm,
              scale=scale, enable_gqa=d["enable_gqa"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="triton",
                    help="reference | torch | triton | native")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    ap.add_argument("--max-print", type=int, default=12)
    args = ap.parse_args()

    if args.impl == "native":
        fn = _native
    else:
        path, entry = IMPLS[args.impl]
        fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tols = gspec["tolerance"]["pointwise_abs"]
    ptols = gspec["tolerance"].get("probs_pointwise_abs", tols)
    rel_tol_special = gspec["tolerance"].get("relative", 1e-3)
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = n_skip = 0
    worst = {"case": "", "err": 0.0, "what": ""}
    print(f"{'case':46s} {'dtype':9s} {'out_err':>10s} {'P_err':>10s} 判定")
    print("-" * 86)
    for i, f in enumerate(idx["files"]):
        d = torch.load(Path(args.golden) / f["file"], weights_only=False)
        name = f["file"].split("/", 1)[-1][:-3]

        if args.impl == "native" and d.get("mask") in ("bool", "bool2d",
                                                       "bool_b1"):
            n_skip += 1
            if n_skip <= args.max_print:
                print(f"{name[:46]:46s} {d['dtype']:9s} {'-':>10s} "
                      f"{'-':>10s} SKIP(bool 直调怪癖)")
            continue

        try:
            out, probs = _run_case(fn, d, d["inputs"], args.device)
        except Exception as e:  # noqa: BLE001 — 逐条报错，最后统一退出码
            n_fail += 1
            print(f"{name[:46]:46s} {d['dtype']:9s} ERROR {str(e)[:60]}")
            continue

        exp = d["expected"]
        e_out = (out.float().cpu() - exp["out"].float()).abs().max().item()
        e_p = (probs.float().cpu() - exp["probs"].float()).abs().max().item()
        special = d.get("case") == "extreme"
        if special:
            rel_o = ((out.float().cpu() - exp["out"].float()).abs()
                     / exp["out"].float().abs().clamp_min(1e-3)).nan_to_num(
                         nan=0.0).max().item()
            rel_p = ((probs.float().cpu() - exp["probs"].float()).abs()
                     / exp["probs"].float().abs().clamp_min(1e-3)).nan_to_num(
                         nan=0.0).max().item()
            ok = rel_o < rel_tol_special and rel_p < rel_tol_special
            shown_o, shown_p = rel_o, rel_p
        else:
            ok = e_out < tols[d["dtype"]] and e_p < ptols[d["dtype"]]
            shown_o, shown_p = e_out, e_p

        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        err = max(shown_o, shown_p)
        if err > worst["err"]:
            worst = {"case": name, "err": err,
                     "what": "rel" if special else "abs"}
        if i < args.max_print or not ok:
            print(f"{name[:46]:46s} {d['dtype']:9s} {shown_o:10.3e} "
                  f"{shown_p:10.3e} {'PASS' if ok else 'FAIL'}")

    print("-" * 86)
    print(f"PASS {n_pass} / FAIL {n_fail} / SKIP {n_skip}"
          f"  worst={worst['err']:.3e}({worst['what']}) @ {worst['case']}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
