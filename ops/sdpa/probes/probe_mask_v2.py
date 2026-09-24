# 重测 flag_gems SDPA mask 路径——mask 在 CPU 构造后搬 NPU，
# 绕开 flag_gems rand kernel（已知 rand.py UB overflow 崩溃点），
# 只测 SDPA 本体对 attn_mask 的处理。
import torch
import torch_npu  # noqa: F401

import flag_gems

torch.manual_seed(0)
DEV = "npu:0"
B, H, Sq, Skv, D = 1, 4, 256, 256, 64


def cpu_ref(q, k, v, m, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float().cpu(), k.float().cpu(), v.float().cpu(),
        attn_mask=m.float().cpu() if m is not None else None,
        is_causal=causal)


# 输入全部在 enable 之前生成好（NPU 上），避免任何生成类算子被接管
qc = torch.randn(B, H, Sq, D) * 0.5
kc = torch.randn(B, H, Skv, D) * 0.5
vc = torch.randn(B, H, Skv, D) * 0.5
mask_bool_cpu = torch.rand(B, H, Sq, Skv) > 0.3
mask_fp_cpu = torch.randn(B, H, Sq, Skv)

flag_gems.enable()

for dt in (torch.float32, torch.float16, torch.bfloat16):
    q, k, v = qc.to(dt).to(DEV), kc.to(dt).to(DEV), vc.to(dt).to(DEV)
    for mask_name, m_cpu in [("bool_mask", mask_bool_cpu),
                             ("float_mask", mask_fp_cpu)]:
        m = m_cpu.to(dt if mask_name == "float_mask" else torch.bool).to(DEV)
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=m, is_causal=False)
            err = (out.float().cpu() - cpu_ref(q, k, v, m, False)).abs().max().item()
            print(f"  [OK]   {str(dt).split('.')[-1]:9s} {mask_name:11s} "
                  f"err={err:.3e}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:130]
            print(f"  [FAIL] {str(dt).split('.')[-1]:9s} {mask_name:11s} "
                  f"{type(e).__name__}: {msg}")

# 附加: GQA 真·不同头数（Hq=8, Hkv=2, enable_gqa）
print("\nGQA (Hq=8 Hkv=2):")
qg_c = torch.randn(1, 8, 128, D) * 0.5
kg_c = torch.randn(1, 2, 128, D) * 0.5
vg_c = torch.randn(1, 2, 128, D) * 0.5
for tag, fn in [
    ("native torch (implicit broadcast)", lambda: torch.nn.functional.
     scaled_dot_product_attention(q, k, v, enable_gqa=True)),
]:
    try:
        q = qg_c.to(torch.float16).to(DEV)
        k = kg_c.to(torch.float16).to(DEV)
        v = vg_c.to(torch.float16).to(DEV)
        out = fn()
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float().cpu(), k.float().cpu(), v.float().cpu(), enable_gqa=True)
        err = (out.float().cpu() - ref).abs().max().item()
        print(f"  [OK]   {tag}: err={err:.3e}")
    except Exception as e:
        print(f"  [FAIL] {tag}: {type(e).__name__}: "
              f"{str(e)[:120].replace(chr(10), ' ')}")
