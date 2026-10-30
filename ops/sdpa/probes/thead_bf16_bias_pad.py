#!/usr/bin/env python3
"""thead-ppu T4 补缺证: bf16 非对齐 bias 的 _aligned_bias 等价性 + bf16 masked 黄金子集回归。

遗留定位（reports/thead-ppu.md §6 关键平台发现 2 / §8 遗留, T1 b29a1ac
message 原话 "bf16 非对齐 kv_pad 未单独验证"）:
  kv_pad 指 **efficient 委托路径** 的 attn_bias 16B stride 对齐修复
  （kernel/backends/thead_ppu.py::_aligned_bias——PLATFORM.md §3c 注明
  "探针里叫 kv_pad"）。此前 fp32/fp16 等价已验（thead_bias_layout.json 9/9），
  bf16 未单独验证；且黄金集 bf16 masked 案例全部 basic S128（连续 bias,
  row stride 256B %16==0）不触发非对齐分支。

本探针（T4 补缺证, 三段）:
  1) bf16 对齐规则复核: 非对齐连续 bias 触发两种报错原文(strideH/无后缀);
  2) bf16 _aligned_bias 实测生产函数 vs 连续 bias 直算(kv_pad 布局+对齐
     连续 bias) + CPU fp32 参考, bf16 容差 1e-2 abs（黄金口径, 收紧于
     inputs_spec bf16 2e-2）内一致;
  3) bf16 masked 黄金子集回归: 黄金 24 组（mbool/mfloat × B1/B2 ×
     c0/c1 × 3 seed）走生产 facade sdpa_triton(cuda:6) 无回退
     （历史 worst 7.812e-3, 规格 2e-2）。

    结果: _aligned_bias 对 bf16 与 fp32/fp16 同为语义等价修复; 黄金 bf16
    masked 走连续 bias 路径确实不触发非对齐分支（黄金数据集形状口径论证
    + 内核错报留证）——T1 §8 遗留第一条收口。

JSON: probes/out/thead_bf16_bias_pad.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = OP_DIR.parents[1]
for p in (str(REPO_ROOT), str(OP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from reference import sdpa_reference  # noqa: E402
from kernel.backends.thead_ppu import _aligned_bias  # noqa: E402
from kernel.triton_level import sdpa_triton  # noqa: E402

TOL_BF16_TASK = 1e-2        # 指令口径: bf16 等价判定线（黄金口径收紧版）
TOL_BF16_SPEC = 2e-2        # inputs_spec.yaml bf16 pointwise_abs
GOLDEN_HIST_WORST = 7.812e-3  # 交付时 bf16 黄金 worst（reports/thead-ppu.md §4）


def make(sq, skv, dt, dev, seed=42, mask_kind="bool"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Hq, Hkv = 4, 4
    q = (torch.randn(1, Hq, sq, 64, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(1, Hkv, skv, 64, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(1, Hkv, skv, 64, generator=g) * 0.5).to(dt).to(dev)
    if mask_kind == "bool":
        m = (torch.rand(1, Hq, sq, skv, generator=g) > 0.3).to(dev)
    else:
        m = (torch.randn(1, Hq, sq, skv, generator=g) * 0.1).to(dt).to(dev)
    return q, k, v, m


def ref_of(q, k, v, m, causal, scale):
    return sdpa_reference(q.cpu(), k.cpu(), v.cpu(),
                          m.cpu() if m is not None else None,
                          0.0, causal, scale, False).to(q.device).float()


def bias_from_mask(m, dt):
    if m.dtype == torch.bool:
        b = torch.zeros_like(m, dtype=dt)
        b.masked_fill_(~m, float("-inf"))
        return b
    return m.to(dt)


def eff(q, k, v, bias, causal, scale):
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, attn_bias=bias, compute_log_sumexp=False, dropout_p=0.0,
        is_causal=causal, scale=scale)[0]


def err_vs_ref(out, ref):
    diff = (out.float() - ref).abs()
    nan_rows = torch.isnan(ref[..., 0])
    if nan_rows.any():
        keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
        diff = diff[keep]
        diff = diff.reshape(-1)
    return diff.max().item() if diff.numel() else 0.0


def try_contig_pad(q, k, v, m, causal, scale):
    """K/V 列 pad + bias 以连续（自然 16B 对齐）缓冲直算——"连续 bias 直算"。

    causal 折叠进 bias（pad 后 Sq != Skv_pad, is_causal 必须关）。
    """
    elem = q.element_size()
    step = 16 // elem
    skv_pad = (k.shape[-2] + step - 1) // step * step
    pad = skv_pad - k.shape[-2]
    k2 = torch.nn.functional.pad(k, (0, 0, 0, pad))
    v2 = torch.nn.functional.pad(v, (0, 0, 0, pad))
    b = torch.nn.functional.pad(bias_from_mask(m, q.dtype),
                                (0, pad), value=float("-inf"))
    if causal:
        c = torch.ones((q.shape[-2], skv_pad), dtype=torch.bool,
                       device=q.device).tril()
        b = b.masked_fill(~c, float("-inf"))
    return eff(q, k2, v2, b, False, scale)


def golden_bf16_masked_files(idx):
    out = []
    for f in idx["files"]:
        name = f["file"]
        if "bfloat16" in name and ("mbool" in name or "mfloat" in name):
            out.append(name)
    return sorted(out)


def main() -> int:
    dev = "cuda:6"
    res = {
        "task": "T4 补缺证: sdpa bf16 非对齐 bias(_aligned_bias/kv_pad) 等价",
        "ts": datetime.now(timezone.utc).isoformat(),
        "device": dev,
        "leftover_refs": {
            "t1_commit": "b29a1ac feat(sdpa): thead-ppu 首算子交付",
            "t1_message_quote": "遗留: bf16 非对齐 kv_pad 未单独验证",
            "report": "ops/sdpa/reports/thead-ppu.md §6#2/§8（bf16 非对齐 stride 的 kv_pad 未单独验证, §6.2）",
            "platform": "ops/sdpa/PLATFORM.md §3c bias 16B stride 对齐行（_aligned_bias, 探针里叫 kv_pad）",
            "meaning": "efficient 委托路径的 attn_bias 16B stride 对齐修复 kernel/backends/thead_ppu.py::_aligned_bias 在 bf16 下未单独验证",
        },
        "rule_check": [],
        "equivalence": [],
        "golden_regression": {"files": [], "worst": 0.0, "worst_case": ""},
        "conclusion": {},
    }
    ok_all = True

    # ---- 1) bf16 对齐规则复核（连续 bias 触发的报错原文） ----
    print("== 1) rule check (bf16 contiguous bias) ==")
    for sq, skv in [(333, 333), (197, 197), (1, 7), (8, 333), (333, 336)]:
        q, k, v, m = make(sq, skv, torch.bfloat16, dev, mask_kind="bool")
        b = bias_from_mask(m, torch.bfloat16)
        head_b = b.stride(1) * 2
        row_b = b.stride(2) * 2
        try:
            eff(q, k, v, b, False, 0.5)
            status = "OK"
        except RuntimeError as e:
            status = str(e).split("aligned")[-1][:24]
        pred = []
        if head_b % 16:
            pred.append("(strideH)")
        if row_b % 16:
            pred.append("(row)")
        print(f"  Sq{sq:5d} Skv{skv:5d} bf16 head={head_b:8d}B "
              f"row={row_b:5d}B → {status!s:24s} pred={''.join(pred)}")
        res["rule_check"].append({"sq": sq, "skv": skv, "dtype": "bfloat16",
                                  "head_bytes": head_b, "row_bytes": row_b,
                                  "observed": status, "predicted": pred})

    # ---- 2) _aligned_bias(bf16 非对齐) vs 连续 bias 直算 vs CPU 参考 ----
    print("== 2) bf16 _aligned_bias vs contiguous direct vs fp32 ref ==")
    rows = [
        (333, 333, "bool", True), (333, 333, "bool", False),
        (333, 333, "float", True), (333, 333, "float", False),
        (197, 197, "bool", True), (197, 197, "float", False),
        (1, 7, "bool", False), (1, 7, "float", False),
        (8, 333, "bool", False), (8, 333, "float", False),
        (333, 336, "bool", False),
    ]
    for sq, skv, mk, causal in rows:
        dt = torch.bfloat16
        q, k, v, m = make(sq, skv, dt, dev, seed=43, mask_kind=mk)
        ref = ref_of(q, k, v, m, causal, 0.5)
        key = f"S{sq}x{skv}/{mk}/causal={int(causal)}"
        row = {}
        # 连续直算: 非对齐必报错（留证）; 对齐形状(333,336)可直接算
        b = bias_from_mask(m, dt)
        try:
            out_c = eff(q, k, v, b, causal, 0.5)
            row["contiguous"] = err_vs_ref(out_c, ref)
        except RuntimeError as e:
            row["contiguous"] = f"ERROR RuntimeError: {str(e)[-60:]}"
        # 生产修复路径: 实测 kernel/backends/thead_ppu.py::_aligned_bias
        out_a = eff(q, k, v, _aligned_bias(b), causal, 0.5)
        row["aligned_bias_vs_ref"] = err_vs_ref(out_a, ref)
        # 连续 bias 直算: K/V pad + 连续对齐 bias（kv_pad 布局）
        out_p = try_contig_pad(q, k, v, m, causal, 0.5)
        row["contig_pad_vs_ref"] = err_vs_ref(out_p, ref)
        # 两路径互差（同一逻辑语义、不同缓冲布局的直碰）
        row["aligned_vs_contig_pad"] = err_vs_ref(out_a, out_p)
        row["pass_1e-2"] = (row["aligned_bias_vs_ref"] <= TOL_BF16_TASK and
                            row["contig_pad_vs_ref"] <= TOL_BF16_TASK and
                            row["aligned_vs_contig_pad"] <= TOL_BF16_TASK)
        ok_all = ok_all and row["pass_1e-2"]
        print(f"  [{'OK' if row['pass_1e-2'] else 'ERR':3s}] {key:24s} "
              f"aligned/contig_pad/pair= {row['aligned_bias_vs_ref']:.3e} / "
              f"{row['contig_pad_vs_ref']:.3e} / {row['aligned_vs_contig_pad']:.3e}"
              f"  contiguous={row['contiguous']}")
        res["equivalence"].append({"key": key, **row})

    # ---- 3) bf16 masked 黄金子集回归（生产 facade 全链路） ----
    print("== 3) golden bf16 masked subset regression (production path) ==")
    spec = json.loads((OP_DIR / "goldendata" / "index.json").read_text())
    files = golden_bf16_masked_files(spec)
    n_pass = 0
    for name in files:
        d = torch.load(OP_DIR / "goldendata" / name)
        ins = [t.to(dev) for t in d["inputs"]]
        q, k, v, m = ins[0], ins[1], ins[2], ins[3]
        out = sdpa_triton(q, k, v, m, 0.0, d["causal"], None, d["Hq"] != d["Hkv"])
        ref = d["expected"].to(dev)
        err = err_vs_ref(out, ref)
        row_pass = err <= TOL_BF16_SPEC
        n_pass += row_pass
        ok_all = ok_all and row_pass
        if err > res["golden_regression"]["worst"]:
            res["golden_regression"]["worst"] = err
            res["golden_regression"]["worst_case"] = name
        res["golden_regression"]["files"].append(
            {"file": name, "abs_err": err, "pass": bool(row_pass)})
        print(f"  [{'OK' if row_pass else 'ERR':3s}] {name:52s} err={err:.3e}")
    res["golden_regression"]["n_files"] = len(files)
    res["golden_regression"]["n_pass"] = n_pass
    res["golden_regression"]["no_regression_vs_worst"] = bool(
        res["golden_regression"]["worst"] <= GOLDEN_HIST_WORST + 1e-6)

    res["conclusion"] = {
        "aligned_bf16_equiv_1e-2": ok_all and n_pass == len(files),
        "contig_direct_error_constrained": "非对齐连续 bias 直算被 kernel 拒（报错已留证）",
        "leftover": "T1 b29a1ac 'bf16 非对齐 kv_pad 未单独验证'",
        "closed": bool(ok_all and n_pass == len(files)
                       and res["golden_regression"]["no_regression_vs_worst"]),
    }
    out_dir = OP_DIR / "probes" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thead_bf16_bias_pad.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("worst golden err:",
          res["golden_regression"]["worst"],
          "(hist", GOLDEN_HIST_WORST, ")")
    print("ALL OK" if res["conclusion"]["closed"] else "FAILURES PRESENT")
    print("JSON:", out_dir / "thead_bf16_bias_pad.json")
    return 0 if res["conclusion"]["closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
