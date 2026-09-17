# 调试2: AutogradPrivateUse1 截流假设——no_grad / 追加 Autograd key 注册
import sys

sys.path.insert(0, "/root/sdpatten-op")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernel.triton_level import sdpa_triton  # noqa: E402

CALLS = {"n": 0}


def aten_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
    CALLS["n"] += 1
    return sdpa_triton(query, key, value, attn_mask, dropout_p,
                       is_causal, scale, enable_gqa)


g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16
dev = "npu:0"
q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)

F = torch.nn.functional

# 实验 1: 仅 PrivateUse1 + no_grad
lib = torch.library.Library("aten", "IMPL")
lib.impl("scaled_dot_product_attention", aten_sdpa, "PrivateUse1")
with torch.no_grad():
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("E1 PrivateUse1 + no_grad:", CALLS["n"])

# 实验 2: 追加 AutogradPrivateUse1
lib2 = torch.library.Library("aten", "IMPL")
lib2.impl("scaled_dot_product_attention", aten_sdpa, "AutogradPrivateUse1")
F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("E2 + AutogradPrivateUse1 (grad mode):", CALLS["n"])
with torch.no_grad():
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("E2 + AutogradPrivateUse1 (no_grad):", CALLS["n"])

# 若 E2 命中: 验证输出正确性 + requires_grad 路径
if CALLS["n"] > 0:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    direct = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    print("hooked == direct:", torch.equal(out, direct))

    # 反向路径: requires_grad=True 时 AutogradPrivateUse1 注册会不会
    # 破坏 torch_npu 的原生反向（我们没有注册 backward）
    qr = q.clone().requires_grad_(True)
    try:
        out_r = F.scaled_dot_product_attention(qr, k, v, is_causal=True)
        loss = out_r.float().sum()
        loss.backward()
        print("grad path OK, q.grad is None:", qr.grad is None)
    except Exception as e:
        print("grad path FAIL:", type(e).__name__, str(e)[:100])

# 实验 3: requires_grad 张量在 E1-only 情况（假设检验）
print("done")
