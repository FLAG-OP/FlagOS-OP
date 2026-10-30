#!/usr/bin/env python3
"""thead-ppu §2b 首测: 厂商委托路径（aten efficient attention）可行性 + 精度。

用法:
  python3 probes/thead_eff_probe.py [--device cuda:6]

输出: 逐案 err vs CPU fp32 参考（判卷口径同 script/check_accuracy）,
      附 enable_gqa / 全遮蔽行 / causal+mask / flash 可用性证据。
JSON: probes/out/thead_eff_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402

from reference import sdpa_reference  # noqa: E402


def make(B, Hq, Hkv, Sq, Skv, D, dt, dev, seed, mask=None, scale=0.5,
         dist="normal"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def rnd(*n):
        return torch.randn(*n, generator=g)
    if dist == "large":
        rnd = lambda *n: torch.rand(*n, generator=g) * 40.0  # noqa: E731
    elif dist == "zeros":
        rnd = lambda *n: torch.zeros(*n)  # noqa: E731
    q = (rnd(B, Hq, Sq, D) * scale).to(dt).to(dev)
    k = (rnd(B, Hkv, Skv, D) * scale).to(dt).to(dev)
    v = (rnd(B, Hkv, Skv, D) * scale).to(dt).to(dev)
    m = None
    if mask == "bool":
        m = (torch.rand(B, Hq, Sq, Skv, generator=g) > 0.3).to(dev)
    elif mask == "float":
        m = (torch.randn(B, Hq, Sq, Skv, generator=g) * 0.1).to(dt).to(dev)
    return q, k, v, m


def vendor_eff(q, k, v, m, causal, scale, enable_gqa):
    """PLATFORM.md §3 p800 先例同款委托（bool→additive bias）。

    PPU torch 2.10 的 efficient schema 无 enable_gqa（实测 8 参数）,
    GQA 由调用方 repeat_interleave 扩 KV（mlu590 overrideable 同款）。
    """
    bias = m
    if bias is not None and bias.dtype == torch.bool:
        bias = torch.zeros_like(bias, dtype=q.dtype)
        bias.masked_fill_(~m, float("-inf"))
    elif bias is not None and bias.dtype != q.dtype:
        bias = bias.to(q.dtype)
    if enable_gqa:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v,
        attn_bias=bias,
        compute_log_sumexp=False,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )[0]
    if m is not None:
        if m.dtype == torch.bool:
            eff = ~m.bool()
        else:
            eff = torch.isneginf(m.float())
        if causal:
            sq, skv = q.shape[-2], k.shape[-2]
            c = torch.ones((sq, skv), dtype=torch.bool,
                           device=q.device).tril()
            eff = eff | ~c
        out = out.masked_fill(eff.all(dim=-1, keepdim=True), float("nan"))
    return out


def run_case(tag, shape, dt, dev, causal, mask, seed=42, scale=0.5,
             dist="normal", enable_gqa=False):
    B, Hq, Hkv, Sq, Skv, D = shape
    q, k, v, m = make(B, Hq, Hkv, Sq, Skv, D, dt, dev, seed, mask, scale,
                      dist)
    try:
        out = vendor_eff(q, k, v, m, causal, scale, enable_gqa)
    except Exception as e:
        return {"case": tag, "dtype": str(dt).split(".")[-1],
                "status": "ERROR", "err": None,
                "note": f"{type(e).__name__}: {str(e)[:160]}"}
    ref = sdpa_reference(
        q.cpu(), k.cpu(), v.cpu(),
        m.cpu() if m is not None else None, 0.0, causal, scale,
        enable_gqa).to(dev).float()
    diff = (out.float() - ref).abs()
    nan_rows = torch.isnan(ref[..., 0])
    if nan_rows.any():
        keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
        diff = diff[keep]
    err = diff.max().item() if diff.numel() else 0.0
    tol = 1e-5 if dt == torch.float32 else 2e-2
    return {"case": tag, "dtype": str(dt).split(".")[-1],
            "status": "PASS" if err < tol else "FAIL", "err": err,
            "tol": tol, "shape": list(shape), "causal": causal,
            "mask": mask}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:6")
    args = ap.parse_args()
    dev = args.device
    res: dict = {"device": dev, "cases": [], "misc": {}}

    props = torch.cuda.get_device_properties(torch.device(dev))
    res["misc"]["device_name"] = props.name
    res["misc"]["aten_eff_exists"] = hasattr(
        torch.ops.aten, "_scaled_dot_product_efficient_attention")

    shapes = [
        ("basic", (1, 4, 4, 128, 128, 64)),
        ("gqa", (1, 8, 2, 256, 256, 64)),
        ("tail100", (1, 4, 4, 100, 100, 64)),
        ("vit-d80", (1, 4, 4, 197, 197, 80)),
        ("tail333", (1, 4, 4, 333, 333, 64)),
        ("tail3072", (1, 4, 4, 3072, 3072, 64)),
        ("tail5000-5120", (1, 4, 4, 5000, 5120, 64)),
        ("decode-sq1", (1, 4, 4, 1, 7, 64)),
        ("d128", (2, 4, 4, 512, 512, 128)),
    ]
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    combos = [
        ("none", False, None),
        ("causal", True, None),
        ("bool", False, "bool"),
        ("float", False, "float"),
        ("causal+bool", True, "bool"),
    ]

    for tag, shape in shapes:
        for dt in dtypes:
            for cname, causal, mask in combos:
                # 大 shape 只跑关键组合，控制首测时长
                if shape[3] >= 3072 and cname not in ("none", "causal",
                                                      "bool"):
                    continue
                # 本探针（2026-10-23 首测口径）跳过非方阵 causal，专门覆盖
                # 该输入的委托可行性见 probes/thead_nonsq_causal_probe.py
                # （PR36 条目 4）——efficient 全 PASS、flash 不生效。
                if causal and shape[3] != shape[4]:
                    continue
                r = run_case(f"{tag}/{cname}", shape, dt, dev, causal, mask,
                             enable_gqa=(shape[1] != shape[2]))
                res["cases"].append(r)
                err = ("%.3e" % r["err"]) if r["err"] is not None else "-"
                print(f"  [{r['status']:5s}] {r['case']:22s} "
                      f"{r['dtype']:9s} err={err} {r.get('note', '')}")

    # extreme（scale=8 饱和 / 全零行 / 大值行）——黄金 extreme 口径
    for dist in ("normal", "zeros", "large"):
        r = run_case(f"extreme/{dist}", (1, 4, 4, 128, 128, 64),
                     torch.float32, dev, True, None, seed=7, scale=8.0,
                     dist=dist)
        res["cases"].append(r)
        err = ("%.3e" % r["err"]) if r["err"] is not None else "-"
        print(f"  [{r['status']:5s}] {r['case']:22s} {r['dtype']:9s} "
              f"err={err} {r.get('note', '')}")

    # 全遮蔽行语义（bool 全 False 行 → 参考 NaN）
    q, k, v, _ = make(1, 4, 4, 64, 64, 64, torch.float16, dev, 5)
    m = torch.ones(1, 4, 64, 64, dtype=torch.bool, device=torch.device(dev))
    m[..., 0, :] = False  # 首行全遮蔽
    out = vendor_eff(q, k, v, m, False, None, False)
    res["misc"]["all_masked_row_nan"] = bool(torch.isnan(out[..., 0, :]).all())
    print("  all-masked row NaN:", res["misc"]["all_masked_row_nan"])

    # flash 可用性（半精度）
    try:
        r = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, True, scale=None)
        res["misc"]["flash_fp16_ok"] = True
        _ = r
    except Exception as e:
        res["misc"]["flash_fp16_ok"] = f"{type(e).__name__}: {str(e)[:120]}"
    print("  flash fp16:", res["misc"]["flash_fp16_ok"])

    n_fail = sum(1 for c in res["cases"] if c["status"] == "FAIL")
    n_err = sum(1 for c in res["cases"] if c["status"] == "ERROR")
    res["summary"] = {"total": len(res["cases"]), "fail": n_fail,
                      "error": n_err}
    print(f"\nTOTAL {len(res['cases'])} FAIL {n_fail} ERROR {n_err}")

    out_dir = OP_DIR / "probes" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thead_eff_probe.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out_dir / "thead_eff_probe.json")
    return 1 if (n_fail or n_err) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
