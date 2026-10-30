#!/usr/bin/env python3
"""sdpa_math thead-ppu 首测探针（任务书 §2b）。

覆盖:
  A. flash/efficient overload schema 与 LSE 可用性（P 物化依赖 LSE）
  B. exact-P 组合精度: flash(maskless)/efficient(masked) O/LSE →
     P=exp(scale·QKᵀ−LSE) vs CPU reference（fp16/bf16 × causal/mask/GQA）
  C. fp32 饱和（extreme scale=8）委托误差——决定 fp32 是否走 ATen 组合
     （T1 sdpa 先例: 委托 abs 2.79e-3 超 1e-5 → fp32 组合）
  D. dispatch key: AutogradCUDA vs CUDA 命中计数（count>0 才算命中;
     每 key 独立子进程）

运行:
  python3 probes/thead_first_probe.py            # A+B+C（单进程, 卡6）
  python3 probes/thead_first_probe.py --key AutogradCUDA
  python3 probes/thead_first_probe.py --key CUDA
JSON: probes/out/thead_first_{exact_p,key_<KEY>}.json
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402

DEV = "cuda:6"
CALLS = {"n": 0}


def _aligned_bias(bias: torch.Tensor) -> torch.Tensor:
    elem = bias.element_size()
    if (bias.stride(2) * elem) % 16 == 0 and (bias.stride(1) * elem) % 16 == 0:
        return bias
    B, H, Sq, Skv = bias.shape
    step = 16 // elem
    skv_pad = (Skv + step - 1) // step * step
    buf = bias.new_zeros(B, H, Sq, skv_pad)
    buf[..., :Skv] = bias
    return buf.as_strided((B, H, Sq, Skv),
                          (H * Sq * skv_pad, Sq * skv_pad, skv_pad, 1))


def _exact_p_from_lse(q, k, out, lse, is_causal, scale, mask=None):
    """P = exp(scale·QKᵀ − LSE)。PPU efficient LSE 按 32 倍数 pad——须切片
    到 Sq（flash LSE 本身即 (B,H,Sq)，切片无副作用）。"""
    sq = q.shape[-2]
    lse_use = lse[..., :sq] if lse.shape[-1] >= sq else lse
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores.mul_(scale if scale is not None else q.shape[-1] ** -0.5)
    scores.sub_(lse_use.unsqueeze(-1).to(scores.dtype)).exp_()
    if is_causal:
        skv = k.shape[-2]
        causal = torch.ones((sq, skv), dtype=torch.bool,
                            device=q.device).tril()
        scores.masked_fill_(~causal, 0)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, 0)
        else:
            scores.masked_fill_(torch.isneginf(mask), 0)
    return out, scores


def _exact_p_flash(q, k, v, is_causal, scale):
    r = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, is_causal, False, scale=scale)
    return _exact_p_from_lse(q, k, r[0], r[1], is_causal, scale)


def _exact_p_efficient(q, k, v, mask, is_causal, scale):
    bias = mask
    if bias is not None and bias.dtype == torch.bool:
        bias = torch.zeros_like(bias, dtype=q.dtype)
        bias.masked_fill_(~mask, float("-inf"))
    elif bias is not None and bias.dtype != q.dtype:
        bias = bias.to(q.dtype)
    if bias is not None:
        bias = _aligned_bias(bias)
    r = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, attn_bias=bias, compute_log_sumexp=True,
        dropout_p=0.0, is_causal=is_causal, scale=scale)
    return _exact_p_from_lse(q, k, r[0], r[1], is_causal, scale, mask=mask)


def _ref(q, k, v, mask, causal, gqa=False):
    from reference import sdpa_math_reference
    return sdpa_math_reference(
        q.cpu(), k.cpu(), v.cpu(),
        mask.cpu() if mask is not None else None, 0.0, causal, None,
        enable_gqa=gqa)


def probe_exact_p() -> dict:
    from reference import sdpa_math_reference  # noqa: F401
    report: dict = {"device": DEV}
    report["device_name"] = torch.cuda.get_device_name(torch.device(DEV))
    report["is_ppu"] = "PPU" in report["device_name"]

    schemas = {}
    for n in ("_scaled_dot_product_flash_attention",
              "_scaled_dot_product_efficient_attention"):
        try:
            schemas[n] = str(getattr(torch.ops.aten, n).default._schema)
        except Exception as exc:
            schemas[n] = f"<missing: {exc}>"
    report["schemas"] = schemas

    # B) exact-P 精度: flash maskless / efficient masked
    cases = []
    torch.manual_seed(20261024)
    specs = [
        # (B, Hq, Hkv, Sq, Skv, D, causal, mask, gqa, tag)
        (1, 4, 4, 128, 128, 64, False, None, False, "basic"),
        (1, 4, 4, 128, 128, 64, True, None, False, "causal"),
        (1, 8, 2, 256, 256, 64, True, None, True, "gqa"),
        (1, 4, 4, 197, 197, 80, True, None, False, "vit-d80"),
        (1, 4, 4, 100, 100, 64, False, "bool", False, "boolmask"),
        (1, 4, 4, 128, 128, 64, False, "float", False, "floatmask"),
    ]
    for (B, Hq, Hkv, Sq, Skv, D, causal, mask_kind, gqa, tag) in specs:
        for dt in (torch.float16, torch.bfloat16):
            g = torch.Generator(device="cpu").manual_seed(42)
            q = (torch.randn(B, Hq, Sq, D, generator=g) * 0.5).to(dt).to(DEV)
            k = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(DEV)
            v = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(DEV)
            mask = None
            if mask_kind == "bool":
                mask = (torch.rand(B, Hq, Sq, Skv, generator=g) > 0.3).to(DEV)
            elif mask_kind == "float":
                mask = (torch.randn(B, Hq, Sq, Skv, generator=g)
                        * 0.1).to(dt).to(DEV)
            qd, kd, vd = q, k, v
            if gqa:
                rep = Hq // Hkv
                kd = k.repeat_interleave(rep, dim=1)
                vd = v.repeat_interleave(rep, dim=1)
            try:
                if mask is None:
                    out, probs = _exact_p_flash(q, kd, vd, causal, None)
                else:
                    out, probs = _exact_p_efficient(q, kd, vd, mask, causal,
                                                    None)
                ref = sdpa_math_reference(
                    q.cpu(), k.cpu(), v.cpu(),
                    mask.cpu() if mask is not None else None, 0.0, causal,
                    None, enable_gqa=gqa)
                eo = (out.float().cpu() - ref[0].float()).abs().max().item()
                ep = (probs.float().cpu() - ref[1].float()).abs().max().item()
                cases.append({
                    "tag": tag, "dtype": str(dt).split(".")[-1],
                    "mask": mask_kind, "out_err": eo, "P_err": ep,
                    "ok_half": max(eo, ep) < 2e-2,
                })
            except Exception as e:  # noqa: BLE001
                cases.append({"tag": tag, "dtype": str(dt).split(".")[-1],
                              "mask": mask_kind,
                              "error": f"{type(e).__name__}: {e}"})
    report["exact_p_half"] = cases

    # C) fp32 委托饱和（extreme: scale=8 大值）——p800/T1 先例判卷
    g = torch.Generator(device="cpu").manual_seed(7)
    q32 = (torch.randn(1, 2, 64, 32, generator=g) * 8.0).to(DEV)
    k32 = (torch.randn(1, 2, 64, 32, generator=g) * 8.0).to(DEV)
    v32 = (torch.randn(1, 2, 64, 32, generator=g) * 0.5).to(DEV)
    ref32 = _ref(q32, k32, v32, None, True)
    fp32_cases = []
    for name, fn in (("flash", _exact_p_flash),):
        try:
            out, probs = fn(q32, k32, v32, True, None)
            eo = (out.cpu() - ref32[0]).abs().max().item()
            ep = (probs.cpu() - ref32[1]).abs().max().item()
            fp32_cases.append({"path": name, "out_err": eo, "P_err": ep,
                               "within_1e5": max(eo, ep) < 1e-5})
        except Exception as e:  # noqa: BLE001
            fp32_cases.append({"path": name,
                               "error": f"{type(e).__name__}: {e}"})
    # 非 extreme fp32 委托精度（对照: T1 记录 3e-7~5e-6）
    q32s = (torch.randn(1, 2, 64, 32, generator=g) * 0.5).to(DEV)
    k32s = (torch.randn(1, 2, 64, 32, generator=g) * 0.5).to(DEV)
    v32s = (torch.randn(1, 2, 64, 32, generator=g) * 0.5).to(DEV)
    ref32s = _ref(q32s, k32s, v32s, None, True)
    try:
        out, probs = _exact_p_flash(q32s, k32s, v32s, True, None)
        eo = (out.cpu() - ref32s[0]).abs().max().item()
        ep = (probs.cpu() - ref32s[1]).abs().max().item()
        fp32_cases.append({"path": "flash-mild", "out_err": eo, "P_err": ep,
                           "within_1e5": max(eo, ep) < 1e-5})
    except Exception as e:  # noqa: BLE001
        fp32_cases.append({"path": "flash-mild",
                           "error": f"{type(e).__name__}: {e}"})
    report["fp32_delegate"] = fp32_cases

    # LSE shape / dtype 抽查
    q = torch.randn(1, 2, 16, 8, device=DEV, dtype=torch.float16)
    r = torch.ops.aten._scaled_dot_product_flash_attention(
        q, q, q, 0.0, True, False, scale=None)
    report["flash_lse_meta"] = {
        "out_shape": list(r[0].shape), "lse_shape": list(r[1].shape),
        "lse_dtype": str(r[1].dtype), "out_contiguous": r[0].is_contiguous(),
    }
    r2 = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, q, q, attn_bias=None, compute_log_sumexp=True,
        dropout_p=0.0, is_causal=True, scale=None)
    report["efficient_lse_meta"] = {
        "out_shape": list(r2[0].shape), "lse_shape": list(r2[1].shape),
        "lse_dtype": str(r2[1].dtype), "out_contiguous": r2[0].is_contiguous(),
    }
    return report


def _counting_math(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, dropout_mask=None, scale=None,
                   enable_gqa=False):
    CALLS["n"] += 1
    return torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
        scale=scale, enable_gqa=enable_gqa)


def probe_key(key: str) -> dict:
    """D) dispatch key 命中计数（先注册计数 impl, 再从**未注册语义**直调——
    注册后 torch.ops 即本实现, 所以命中面=调用面）。"""
    res: dict = {"dispatch_key": key, "device": DEV, "probes": {}}
    lib = torch.library.Library("aten", "IMPL")
    # 本探针只测"该 key 下 Python impl 是否被 dispatcher 调用"——
    # impl 内部转发原生会递归, 故用 flag 标记后走 torch_level 组合。
    from kernel.torch_level import sdpa_math_torch

    def impl_fn(query, key_, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, dropout_mask=None, scale=None,
                enable_gqa=False):
        CALLS["n"] += 1
        return sdpa_math_torch(query, key_, value, attn_mask, dropout_p,
                               is_causal, dropout_mask, scale=scale,
                               enable_gqa=enable_gqa)

    lib.impl("_scaled_dot_product_attention_math", impl_fn, key)
    res["has_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_scaled_dot_product_attention_math", key)

    g = torch.Generator(device="cpu").manual_seed(3)

    def mk(rg, dt=torch.float32):
        q = (torch.randn(1, 2, 16, 8, generator=g) * 0.5).to(dt).to(
            DEV).requires_grad_(rg)
        k = (torch.randn(1, 2, 16, 8, generator=g) * 0.5).to(dt).to(
            DEV).requires_grad_(rg)
        v = (torch.randn(1, 2, 16, 8, generator=g) * 0.5).to(dt).to(
            DEV).requires_grad_(rg)
        return q, k, v

    def call(q, k, v):
        return torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, True, None)

    q, k, v = mk(False)
    CALLS["n"] = 0
    _ = call(q, k, v)
    res["probes"]["grad_no_rg"] = CALLS["n"]

    q, k, v = mk(True)
    CALLS["n"] = 0
    out, probs = call(q, k, v)
    res["probes"]["grad_rg_has_grad_fn"] = out.grad_fn is not None
    try:
        (out.float().sum() + probs.float().sum()).backward()
        bwd_ok = bool(q.grad is not None and torch.isfinite(q.grad).all())
    except Exception as e:  # noqa: BLE001
        bwd_ok = f"{type(e).__name__}: {str(e)[:120]}"
    res["probes"]["grad_rg"] = CALLS["n"]
    res["probes"]["grad_rg_backward"] = bwd_ok

    q, k, v = mk(False)
    CALLS["n"] = 0
    with torch.no_grad():
        _ = call(q, k, v)
    res["probes"]["no_grad"] = CALLS["n"]

    q, k, v = mk(False)
    CALLS["n"] = 0
    with torch.inference_mode():
        _ = call(q, k, v)
    res["probes"]["inference_mode"] = CALLS["n"]

    q, k, v = mk(False)
    CALLS["n"] = 0
    _ = torch.ops.aten._scaled_dot_product_attention_math.default(
        q, k, v, None, 0.0, True, None)
    res["probes"]["aten_direct"] = CALLS["n"]

    hits = [k_ for k_, v_ in res["probes"].items()
            if isinstance(v_, int) and v_ > 0]
    res["hit_probes"] = hits
    print(f"[{key}] has_kernel={res['has_kernel']} probes={res['probes']}")
    print(f"[{key}] HIT: {hits or 'NONE'}")
    return res


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--key":
        key = sys.argv[2]
        res = probe_key(key)
        out = OP_DIR / "probes" / "out" / f"thead_first_key_{key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("JSON:", out)
        return 0 if res["hit_probes"] else 1

    res = probe_exact_p()
    out = OP_DIR / "probes" / "out" / "thead_first_exact_p.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
