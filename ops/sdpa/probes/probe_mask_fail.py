# 抓 flag_gems SDPA mask 路径完整编译错误（报告证据，函数级定位）
import traceback

import torch
import torch_npu  # noqa: F401

import flag_gems

flag_gems.enable()
torch.manual_seed(0)
B, H, Sq, Skv, D = 1, 4, 256, 256, 64
q = torch.randn(B, H, Sq, D, dtype=torch.float16, device="npu:0") * 0.5
k = torch.randn(B, H, Skv, D, dtype=torch.float16, device="npu:0") * 0.5
v = torch.randn(B, H, Skv, D, dtype=torch.float16, device="npu:0") * 0.5
m = torch.rand(B, H, Sq, Skv, dtype=torch.float16, device="npu:0") > 0.3
try:
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=m, is_causal=False)
    print("UNEXPECTED OK", out.shape)
except Exception:
    tb = traceback.format_exc()
    print(tb)
    print("---- 关键行 ----")
    for ln in tb.splitlines():
        if ("File \"" in ln and "flag_gems" in ln) or "Error" in ln:
            print(ln.strip()[:200])
