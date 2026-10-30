#!/usr/bin/env python3
"""thead-ppu §2b 首测(3): A1 dispatch key 命中实测（call count>0 才算命中）。

用法:
  python3 probes/thead_reg_probe.py AutogradCUDA
  python3 probes/thead_reg_probe.py CUDA
每个 key 必须独立进程（torch.library 注册不可撤销, 见 test/op_level.py 注）。
JSON: probes/out/thead_reg_<key>.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402


CALLS = {"n": 0}


def counting_impl(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False,
                  *args, **kwargs):
    """计数 wrapper: 命中后走厂商 efficient 直调（不回 F.sdpa, 防递归）。"""
    CALLS["n"] += 1
    bias = attn_mask
    if bias is not None and bias.dtype == torch.bool:
        bias = torch.zeros_like(bias, dtype=query.dtype)
        bias.masked_fill_(~attn_mask, float("-inf"))
    if enable_gqa:
        rep = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        query, key, value, attn_bias=bias,
        compute_log_sumexp=torch.is_grad_enabled(),
        dropout_p=dropout_p, is_causal=is_causal, scale=scale)[0]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    key = sys.argv[1]
    dev = "cuda:6"
    res: dict = {"dispatch_key": key, "device": dev, "probes": {}}

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("scaled_dot_product_attention", counting_impl, key)
    res["has_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::scaled_dot_product_attention", key)

    g = torch.Generator(device="cpu").manual_seed(3)
    def mk(requires_grad):
        q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(
            torch.float16).to(dev).requires_grad_(requires_grad)
        k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(
            torch.float16).to(dev).requires_grad_(requires_grad)
        v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(
            torch.float16).to(dev).requires_grad_(requires_grad)
        return q, k, v

    F = torch.nn.functional

    # 1) grad mode, no requires_grad（kernel/op 层测试口径）
    q, k, v = mk(False)
    CALLS["n"] = 0
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    res["probes"]["grad_no_rg"] = CALLS["n"]

    # 2) grad mode + requires_grad + backward
    q, k, v = mk(True)
    CALLS["n"] = 0
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    has_grad_fn = out.grad_fn is not None
    try:
        out.float().sum().backward()
        bwd_ok = bool(q.grad is not None and torch.isfinite(q.grad).all())
    except Exception as e:
        bwd_ok = f"{type(e).__name__}: {str(e)[:120]}"
    res["probes"]["grad_rg"] = CALLS["n"]
    res["probes"]["grad_rg_grad_fn"] = has_grad_fn
    res["probes"]["grad_rg_backward"] = bwd_ok

    # 3) torch.no_grad（framework 层测试口径）
    q, k, v = mk(False)
    CALLS["n"] = 0
    with torch.no_grad():
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    res["probes"]["no_grad"] = CALLS["n"]

    # 4) inference_mode（known-issues #19 教训: 可能零命中）
    q, k, v = mk(False)
    CALLS["n"] = 0
    with torch.inference_mode():
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    res["probes"]["inference_mode"] = CALLS["n"]

    # 5) 直调 aten（绕 F 包装）
    q, k, v = mk(False)
    CALLS["n"] = 0
    _ = torch.ops.aten.scaled_dot_product_attention.default(
        q, k, v, None, 0.0, True)
    res["probes"]["aten_direct"] = CALLS["n"]

    hits = [k for k, v_ in res["probes"].items()
            if isinstance(v_, int) and v_ > 0]
    res["hit_probes"] = hits
    print(f"[{key}] has_kernel={res['has_kernel']} probes={res['probes']}")
    print(f"[{key}] HIT: {hits or 'NONE'}")

    out_dir = OP_DIR / "probes" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"thead_reg_{key}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out_dir / f"thead_reg_{key}.json")
    return 0 if hits else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
