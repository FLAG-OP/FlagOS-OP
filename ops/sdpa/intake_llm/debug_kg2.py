# 最小化定位: 单块化(BLOCK_N=128)后 S=128 是否归零 → 区分
# "跨块状态 bug" vs "PATCH-2 循环边界 bug"
import sys

sys.path.insert(0, "/root/sdpatten-op")
sys.path.insert(0, "/root/sdpatten-op/intake_llm")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

import kernelgen_triton as kgt
from reference import sdpa_reference

g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16
q = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")
k = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")
v = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")

# monkey-patch wrapper 的 BLOCK_N: 重新实现一个调用（直接改模块级不可行，
# wrapper 内部硬编码 64 → 复制调用逻辑，BLOCK_N=128 单块）
import math
import triton

out = torch.empty_like(q)
B, H, Sq, D = q.shape
scale = 1.0 / math.sqrt(D)
BLOCK_K = 64
BLOCK_M = 64
BLOCK_N = 128  # ← 单块覆盖整个 K 序列
grid = (triton.cdiv(Sq, BLOCK_M), B * H)
kgt._sdpa_kernel[grid](
    q, k, v, out, q,
    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    0, 0, Sq, Sq, D, scale, 0.0, 0,
    HAS_MASK=False, IS_CAUSAL=False, HAS_DROPOUT=False,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
)
ref = sdpa_reference(q, k, v, None, 0.0, False, None, False).float()
print("BLOCK_N=128 单块 S=128 err:",
      (out.float() - ref).abs().max().item())
print("→ 若≈0: 跨块 online-softmax 状态 bug; 若仍错: 块内计算 bug")
