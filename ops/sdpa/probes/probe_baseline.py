# 现状基线探测: 原生 torch SDPA 与 flag_gems SDPA 在本机 NPU 的可用性
# 运行: python3 probe_baseline.py
import traceback

import torch
import torch_npu  # noqa: F401

torch.manual_seed(0)
DEV = "npu:0"

CASes = [
    # (name, B, H, Sq, Skv, D, dtype, is_causal, attn_mask)
    ("fp32_small_noCausal", 2, 4, 128, 128, 64, torch.float32, False, None),
    ("fp32_small_causal", 2, 4, 128, 128, 64, torch.float32, True, None),
    ("fp16_small_causal", 2, 4, 128, 128, 64, torch.float16, True, None),
    ("bf16_small_causal", 2, 4, 128, 128, 64, torch.bfloat16, True, None),
    ("fp16_long_causal", 1, 8, 1024, 1024, 64, torch.float16, True, None),
    ("fp16_GQA_causal", 1, 8, 512, 512, 64, torch.float16, True, None),
    ("fp16_masked", 1, 4, 256, 256, 64, torch.float16, False, "mask"),
]


def make_case(B, H, Sq, Skv, D, dt, causal, mask):
    q = torch.randn(B, H, Sq, D, dtype=dt, device=DEV) * 0.5
    k = torch.randn(B, H, Skv, D, dtype=dt, device=DEV) * 0.5
    v = torch.randn(B, H, Skv, D, dtype=dt, device=DEV) * 0.5
    m = None
    if mask == "mask":
        m = torch.rand(B, H, Sq, Skv, dtype=dt, device=DEV) > 0.3
    return q, k, v, m, causal


print("=" * 78)
print("A. 原生 torch SDPA (aten dispatch → NPU 实现)")
print("=" * 78)
for name, B, H, Sq, Skv, D, dt, causal, mask in CASes:
    try:
        q, k, v, m, causal = make_case(B, H, Sq, Skv, D, dt, causal, mask)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=causal)
        # CPU fp32 参考对比
        qc, kc, vc = q.float().cpu(), k.float().cpu(), v.float().cpu()
        mc = m.float().cpu() if m is not None else None
        ref = torch.nn.functional.scaled_dot_product_attention(
            qc, kc, vc, attn_mask=mc, is_causal=causal)
        err = (out.float().cpu() - ref).abs().max().item()
        print(f"  [OK] {name:24s} max_err_vs_cpufp32={err:.3e}")
    except Exception as e:
        print(f"  [FAIL] {name:24s} {type(e).__name__}: {str(e)[:110]}")

print()
print("=" * 78)
print("B. flag_gems SDPA (enable 后 aten 路径)")
print("=" * 78)
import flag_gems  # noqa: E402

flag_gems.enable()
for name, B, H, Sq, Skv, D, dt, causal, mask in CASes:
    try:
        q, k, v, m, causal = make_case(B, H, Sq, Skv, D, dt, causal, mask)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=causal)
        qc, kc, vc = q.float().cpu(), k.float().cpu(), v.float().cpu()
        mc = m.float().cpu() if m is not None else None
        ref = torch.nn.functional.scaled_dot_product_attention(
            qc, kc, vc, attn_mask=mc, is_causal=causal)
        err = (out.float().cpu() - ref).abs().max().item()
        print(f"  [OK] {name:24s} max_err_vs_cpufp32={err:.3e}")
    except Exception as e:
        print(f"  [FAIL] {name:24s} {type(e).__name__}: {str(e)[:110]}")

print()
print("=" * 78)
print("C. flag_gems SDPA 直调（不经 aten，ScaleDotProductAttention.apply）")
print("=" * 78)
for name, B, H, Sq, Skv, D, dt, causal, mask in CASes[:6]:
    try:
        from flag_gems.ops.attention import scaled_dot_product_attention_forward
        q, k, v, m, causal = make_case(B, H, Sq, Skv, D, dt, causal, mask)
        o, M = scaled_dot_product_attention_forward(
            q, k, v, m, 0.0, causal, None, False)
        qc, kc, vc = q.float().cpu(), k.float().cpu(), v.float().cpu()
        mc = m.float().cpu() if m is not None else None
        ref = torch.nn.functional.scaled_dot_product_attention(
            qc, kc, vc, attn_mask=mc, is_causal=causal)
        err = (o.float().cpu() - ref).abs().max().item()
        print(f"  [OK] {name:24s} max_err_vs_cpufp32={err:.3e}")
    except Exception as e:
        print(f"  [FAIL] {name:24s} {type(e).__name__}: {str(e)[:110]}")

print()
print("D. GQA case 的 q/k head 数 (第6条 fp16_GQA_causal Sq=Skv=512):")
print("   该 case H=8/8 → 其实是 MHA。真正 GQA 需要 Hq != Hkv，flag_gems 需 enable_gqa=True")
EOF_MARKER = None
