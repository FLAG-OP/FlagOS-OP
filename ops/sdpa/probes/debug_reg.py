# 调试: 为什么 PrivateUse1 的 python IMPL 注册没拦截 F.sdpa
import sys

sys.path.insert(0, "/root/sdpatten-op")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

CALLS = {"n": 0}

from kernel.triton_level import sdpa_triton  # noqa: E402


def aten_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
    CALLS["n"] += 1
    return sdpa_triton(query, key, value, attn_mask, dropout_p,
                       is_causal, scale, enable_gqa)


lib = torch.library.Library("aten", "IMPL")
lib.impl("scaled_dot_product_attention", aten_sdpa, "PrivateUse1")
print("registered, lib:", lib)

g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16
dev = "npu:0"
q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)

# 1) 直调 aten op（绕过 F 包装）
o1 = torch.ops.aten.scaled_dot_product_attention.default(
    q, k, v, None, 0.0, True)
print("aten direct call count:", CALLS["n"])

# 2) F 包装
o2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print("F.sdpa call count:", CALLS["n"])

# 3) 查看 python 注册后 dispatch dump 里 SDPA 的 PrivateUse1 项
tbl = torch._C._dispatch_dump("aten::scaled_dot_product_attention")
for ln in tbl.splitlines():
    if "PrivateUse1" in ln and "Autograd" not in ln and "Autocast" not in ln:
        print("dump:", ln.strip()[:160])

# 4) has_kernel 检查
print("has PrivateUse1:",
      torch._C._dispatch_has_kernel_for_dispatch_key(
          "aten::scaled_dot_product_attention", "PrivateUse1"))

# 5) 试 CompositeExplicitAutograd / Autograd key 对照
print("done")
