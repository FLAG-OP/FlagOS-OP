# 验证 flag_gems.enable() 后 SDPA 是否真的接管 aten 路径（PrivateUse1）
import torch
import torch_npu  # noqa: F401

import flag_gems

torch.manual_seed(0)
DEV = "npu:0"
B, H, S, D = 1, 4, 128, 64

# enable 前生成输入（保证两次调用输入完全相同）
q_c = torch.randn(B, H, S, D) * 0.5
k_c = torch.randn(B, H, S, D) * 0.5
v_c = torch.randn(B, H, S, D) * 0.5

q, k, v = (t.to(torch.float16).to(DEV) for t in (q_c, k_c, v_c))

out_native = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True).clone()

# 直调 flag_gems 前向（确定 Triton kernel 输出）
from flag_gems.ops.attention import scaled_dot_product_attention_forward
o_gems, M = scaled_dot_product_attention_forward(q, k, v, None, 0.0, True,
                                                 None, False)

flag_gems.enable()

out_after = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True)

print("native vs gems-direct max diff:",
      (out_native.float() - o_gems.float()).abs().max().item())
print("after-enable vs native    max diff:",
      (out_after.float() - out_native.float()).abs().max().item())
print("after-enable vs gems      max diff:",
      (out_after.float() - o_gems.float()).abs().max().item())

# 直接看 dispatcher 里 SDPA 的 PrivateUse1 实现
import torch._C as C
try:
    d = torch.ops.aten.scaled_dot_product_attention.default._schema
    print("schema:", d)
except Exception as e:
    print("schema probe err:", e)

# 用 torch.library 检查注册表
try:
    has = torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::scaled_dot_product_attention", "PrivateUse1")
    print("PrivateUse1 kernel registered for SDPA:", has)
except Exception as e:
    print("dispatch probe err:", e)

# Python 层注册检测: TORCH_LIBRARY_IMPL 记录
tbl = torch._C._dispatch_dump("aten::scaled_dot_product_attention")
print("dispatch table:\n", tbl)
