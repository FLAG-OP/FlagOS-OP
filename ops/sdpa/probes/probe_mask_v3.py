# 修正参考（bool mask 保持 bool 语义）后重测 SDPA mask 路径
import torch
import torch_npu  # noqa: F401

import flag_gems

torch.manual_seed(0)
DEV = "npu:0"
B, H, Sq, Skv, D = 1, 4, 256, 256, 64

qc = torch.randn(B, H, Sq, D) * 0.5
kc = torch.randn(B, H, Skv, D) * 0.5
vc = torch.randn(B, H, Skv, D) * 0.5
mb = torch.rand(B, H, Sq, Skv) > 0.3          # bool
mf = torch.randn(B, H, Sq, Skv)               # float additive


def cpu_ref(q, k, v, m, causal):
    # bool mask 保持 bool（遮蔽语义）；float mask 升 fp32（加性语义）
    if m is None:
        mm = None
    elif m.dtype == torch.bool:
        mm = m.cpu()
    else:
        mm = m.float().cpu()
    return torch.nn.functional.scaled_dot_product_attention(
        q.float().cpu(), k.float().cpu(), v.float().cpu(),
        attn_mask=mm, is_causal=causal)


def run_one(tag, sdpa):
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        q, k, v = (t.to(dt).to(DEV) for t in (qc, kc, vc))
        for mn, m in [("bool", mb), ("float", mf)]:
            mm = m.to(DEV) if mn == "bool" else m.to(dt).to(DEV)
            out = sdpa(q, k, v, attn_mask=mm, is_causal=False)
            err = (out.float().cpu() - cpu_ref(q, k, v, mm, False)
                   ).abs().max().item()
            print(f"  {tag:11s} {str(dt).split('.')[-1]:9s} {mn:5s} "
                  f"err={err:.3e}")


print("== 原生（未 enable）==")
run_one("native", torch.nn.functional.scaled_dot_product_attention)
flag_gems.enable()
print("== flag_gems enable ==")
run_one("flaggems", torch.nn.functional.scaled_dot_product_attention)
