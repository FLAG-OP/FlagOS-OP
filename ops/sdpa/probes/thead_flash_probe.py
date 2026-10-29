#!/usr/bin/env python3
"""thead §h 性能注记探针 v2: 各厂商 SDPA overload 与 F.sdpa 原生对照。

背景: v1 实测 flash 直调与原生同速, 但集成后同进程长 warmup 复测出现
分化（原生 0.018ms vs flash 直调 0.069ms, 1k/d128 causal fp16）——怀疑
原生 F.sdpa 在 PPU 走的是 vendor 注册的 overrideable fused 后端而非
通用 flash 包装。本探针对四个入口同口径计时（warmup30/it50, 整圈
synchronize）, 并核 overrideable 的 maskless/masked 精度。
JSON 落 probes/out/thead_flash_probe.json（v2 覆盖写）。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "out" / "thead_flash_probe.json"
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:6"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import sdpa_reference  # noqa: E402


def bench(fn, warmup=30, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> int:
    torch.cuda.set_device(torch.device(DEV))
    report: dict = {"device": DEV, "note": "v2",
                    "gpu_name": torch.cuda.get_device_name()}

    B, H, S, D = 1, 16, 1024, 128
    torch.manual_seed(20261023)
    q = (torch.randn(B, H, S, D) * 0.03).half().to(DEV)
    k = (torch.randn(B, H, S, D) * 0.03).half().to(DEV)
    v = (torch.randn(B, H, S, D) * 0.03).half().to(DEV)
    ref = sdpa_reference(q, k, v, None, 0.0, True, None, False)

    import torch.nn.functional as F

    # 1) overload schema 盘点（flash 无 attn_mask; cudnn 在 PPU 上报错——
    #    两者都是 PLATFORM.md thead 节要引用的原文证据）
    schemas = {}
    for n in ("_scaled_dot_product_flash_attention",
              "_scaled_dot_product_efficient_attention",
              "_scaled_dot_product_fused_attention_overrideable",
              "_scaled_dot_product_cudnn_attention"):
        try:
            schemas[n] = str(getattr(torch.ops.aten, n).default._schema)
        except Exception as exc:
            schemas[n] = f"<missing: {exc}>"
    report["schemas"] = schemas
    try:
        torch.ops.aten._scaled_dot_product_cudnn_attention(
            q, k, v, attn_bias=None, compute_log_sumexp=False,
            dropout_p=0.0, is_causal=True, return_debug_mask=False,
            scale=None)
        report["cudnn_call"] = "unexpectedly succeeded"
    except Exception as exc:
        report["cudnn_call"] = f"{type(exc).__name__}: {exc}"

    def flash():
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, True, False, scale=None)[0]

    def efficient():
        return torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, attn_bias=None, compute_log_sumexp=False,
            dropout_p=0.0, is_causal=True, scale=None)[0]

    def overrideable():
        return torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q, k, v, attn_bias=None, dropout_p=0.0, is_causal=True,
            return_debug_mask=False, scale=None)[0]

    report["ms_native"] = bench(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    for name, fn in (("flash", flash), ("efficient", efficient),
                     ("overrideable", overrideable)):
        try:
            out = fn()
            report[f"{name}_err_vs_gold"] = (
                out.float() - ref.float()).abs().max().item()
            report[f"ms_{name}"] = bench(fn)
        except Exception as exc:
            report[f"{name}_call"] = f"{type(exc).__name__}: {exc}"

    # overrideable 带 mask 精度（bool mask → additive bias, 与生产路径同款转换）
    torch.manual_seed(7)
    mask = torch.rand(B, H, S, S, device=DEV) > 0.3
    ref_m = sdpa_reference(q, k, v, mask, 0.0, False, None, False)
    try:
        bias = torch.zeros_like(mask, dtype=q.dtype)
        bias.masked_fill_(~mask, float("-inf"))
        out_m = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q, k, v, attn_bias=bias, dropout_p=0.0, is_causal=False,
            return_debug_mask=False, scale=None)[0]
        kept = mask.any(dim=-1, keepdim=True)
        err_m = (out_m.float() - ref_m.float()).abs()[kept].max().item()
        report["overrideable_mask_err_vs_gold"] = err_m
        report["overrideable_mask_ok"] = err_m < 2e-2
    except Exception as exc:
        report["overrideable_mask_call"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
