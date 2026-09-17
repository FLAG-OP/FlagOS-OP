# 独立进程: 采集未注册状态下的原生 SDPA 前向+梯度（op 层对照用）
# 输出: NATIVE_GRAD {"dq": [...], "dk": [...], "dv": [...]}（fp32 值）
import sys

sys.path.insert(0, "/root/sdpatten-op")
import json  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16
dev = "npu:0"
mk = lambda n: (torch.randn(*n, generator=g) * 0.5)  # noqa: E731
q = mk((1, 4, 128, 64)).to(dt).to(dev).requires_grad_(True)
k = mk((1, 4, 128, 64)).to(dt).to(dev).requires_grad_(True)
v = mk((1, 4, 128, 64)).to(dt).to(dev).requires_grad_(True)

out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True)
loss = out.float().sum()
loss.backward()
print("NATIVE_GRAD", json.dumps({
    "dq": q.grad.float().cpu().tolist(),
    "dk": k.grad.float().cpu().tolist(),
    "dv": v.grad.float().cpu().tolist(),
}))
