#!/usr/bin/env python3
"""thead-ppu §2b 首测(2): efficient attention 的 attn_bias 对齐约束与规避布局。

已知现象（thead_eff_probe）: 某些 (Sq,Skv,dtype) 下
  RuntimeError: attn_bias is not correctly aligned [(strideH)]
推定规则（对连续 bias, elem=元素字节数）:
  check1 (strideH): (stride(1) * elem) % 16 == 0
  check2         : (stride(2) * elem) % 16 == 0
本探针:
  1) 扫描 (Sq,Skv,dtype) 复核规则;
  2) 试三种布局: 连续 / as_strided 行距填充(形状不变) / K,V 列填充;
  3) 每种与 CPU fp32 参考比对（语义等价才可用）。

JSON: probes/out/thead_bias_layout.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402

from reference import sdpa_reference  # noqa: E402


def make(sq, skv, dt, dev, seed=42, mask_kind="bool", gqa=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Hq, Hkv = (8, 2) if gqa else (4, 4)
    q = (torch.randn(1, Hq, sq, 64, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(1, Hkv, skv, 64, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(1, Hkv, skv, 64, generator=g) * 0.5).to(dt).to(dev)
    if mask_kind == "bool":
        m = (torch.rand(1, Hq, sq, skv, generator=g) > 0.3).to(dev)
    else:
        m = (torch.randn(1, Hq, sq, skv, generator=g) * 0.1).to(dt).to(dev)
    return q, k, v, m


def ref_of(q, k, v, m, causal, scale, gqa):
    return sdpa_reference(q.cpu(), k.cpu(), v.cpu(),
                          m.cpu() if m is not None else None,
                          0.0, causal, scale, gqa).to(q.device).float()


def bias_from_mask(m, dt):
    if m.dtype == torch.bool:
        b = torch.zeros_like(m, dtype=dt)
        b.masked_fill_(~m, float("-inf"))
        return b
    return m.to(dt)


def align(x: int, elem: int) -> int:
    """返回 ≥x 且满足 (y*elem)%16==0 的最小 y。"""
    step = 16 // elem
    if x * elem % 16 == 0:
        return x
    return (x + step - 1) // step * step


def eff(q, k, v, bias, causal, scale):
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, attn_bias=bias, compute_log_sumexp=False, dropout_p=0.0,
        is_causal=causal, scale=scale)[0]


def try_contiguous(q, k, v, m, causal, scale):
    return eff(q, k, v, bias_from_mask(m, q.dtype), causal, scale)


def try_as_strided(q, k, v, m, causal, scale):
    """逻辑形状不变, 仅把 bias 的行距/头距填充到 16B 对齐（K-stride 仍 1）。

    行距 = skv_pad*elem ≡ 0 (mod 16) ⇒ 头距 = Sq*行距 亦 ≡ 0。
    """
    elem = q.dtype.itemsize
    b = bias_from_mask(m, q.dtype)
    B, H, Sq, Skv = b.shape
    skv_pad = align(Skv, elem)
    if skv_pad == Skv:
        return eff(q, k, v, b, causal, scale)
    buf = torch.zeros(B, H, Sq, skv_pad, dtype=b.dtype, device=b.device)
    buf[..., :Skv] = b
    view = buf.as_strided((B, H, Sq, Skv),
                          (H * Sq * skv_pad, Sq * skv_pad, skv_pad, 1))
    return eff(q, k, v, view, causal, scale)


def try_kv_pad(q, k, v, m, causal, scale):
    """K/V 列填充 + bias 列 -inf（语义: 追加 key 的 softmax 权重为 0）。"""
    elem = q.dtype.itemsize
    step = 16 // elem
    B, Hq, Sq, Skv = q.shape[0], q.shape[1], q.shape[-2], k.shape[-2]
    skv_pad = (Skv + step - 1) // step * step
    if skv_pad == Skv and (Sq * Skv * elem) % 16 == 0:
        return eff(q, k, v, bias_from_mask(m, q.dtype), causal, scale)
    pad = skv_pad - Skv
    k2 = torch.nn.functional.pad(k, (0, 0, 0, pad))
    v2 = torch.nn.functional.pad(v, (0, 0, 0, pad))
    b = torch.nn.functional.pad(bias_from_mask(m, q.dtype),
                                (0, pad), value=float("-inf"))
    # causal 折叠进 bias（padding 后 Sq != Skv_pad, 不再用 is_causal）
    if causal:
        c = torch.ones((Sq, skv_pad), dtype=torch.bool, device=q.device)
        c = c.tril()
        b = b.masked_fill(~c, float("-inf"))
    out = eff(q, k2, v2, b, False, scale)
    return out


def report(tag, out, ref):
    diff = (out.float() - ref).abs()
    nan_rows = torch.isnan(ref[..., 0])
    if nan_rows.any():
        keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
        diff = diff[keep]
    err = diff.max().item() if diff.numel() else 0.0
    print(f"  [{('OK' if err < 1e-5 else 'ERR'):3s}] {tag:52s} "
          f"err={err:.3e}")
    return err


def main() -> int:
    dev = "cuda:6"
    res = {"device": dev, "rule_check": [], "layout": {}}

    # 1) 规则复核: 用 as_strided 构造已知 stride 的 bias, 看报错分支
    print("== rule check (contiguous bias) ==")
    for sq, skv, dt in [(333, 333, torch.float32), (100, 100, torch.float16),
                        (197, 197, torch.float32), (1, 7, torch.float32),
                        (100, 100, torch.float32), (128, 128, torch.float16),
                        (100, 128, torch.float16), (127, 128, torch.float32),
                        (126, 128, torch.float16)]:
        q, k, v, m = make(sq, skv, dt, dev, mask_kind="bool")
        b = bias_from_mask(m, dt)
        head_b = b.stride(1) * dt.itemsize
        row_b = b.stride(2) * dt.itemsize
        try:
            eff(q, k, v, b, False, None)
            status = "OK"
        except RuntimeError as e:
            status = str(e).split("aligned")[-1][:24]
        pred = []
        if head_b % 16:
            pred.append("(strideH)")
        if row_b % 16:
            pred.append("(row)")
        print(f"  Sq{sq:4d} Skv{skv:4d} {str(dt)[6:]:10s} "
              f"head={head_b:6d}B row={row_b:6d}B → {status!s:26s} "
              f"pred={''.join(pred)}")
        res["rule_check"].append(
            {"sq": sq, "skv": skv, "dtype": str(dt), "head_bytes": head_b,
             "row_bytes": row_b, "observed": status, "predicted": pred})

    # 2) 布局方案 vs 参考
    print("== layout variants vs CPU fp32 reference ==")
    for sq, skv, dt, causal in [(333, 333, torch.float32, True),
                                (333, 333, torch.float16, True),
                                (197, 197, torch.float32, True),
                                (100, 100, torch.float16, True),
                                (1, 7, torch.float32, False),
                                (100, 100, torch.bfloat16, False),
                                (333, 333, torch.float32, False)]:
        for mask_kind in ("bool", "float"):
            q, k, v, m = make(sq, skv, dt, dev, mask_kind=mask_kind)
            ref = ref_of(q, k, v, m, causal, 0.5, False)
            key = f"S{sq}x{skv}/{str(dt)[6:]}/{mask_kind}/causal={int(causal)}"
            row = {}
            for name, fn in [("contiguous", try_contiguous),
                             ("as_strided", try_as_strided),
                             ("kv_pad", try_kv_pad)]:
                try:
                    out = fn(q, k, v, m, causal, 0.5)
                    row[name] = report(f"{key} {name}", out, ref)
                except Exception as e:
                    row[name] = f"ERROR {type(e).__name__}: {str(e)[:100]}"
                    print(f"  [ERR] {key} {name}: {row[name]}")
            res["layout"][key] = row

    out_dir = OP_DIR / "probes" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thead_bias_layout.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out_dir / "thead_bias_layout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
