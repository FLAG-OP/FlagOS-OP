# 终极定位: S=128 双块, 逐 M 块(64 行一组)查误差 + 与"只用第一个K块"
# "只用第二个K块"的假设输出比对 → 确定哪个块的贡献错了
import sys

sys.path.insert(0, "/root/sdpatten-op")
sys.path.insert(0, "/root/sdpatten-op/intake_llm")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernelgen_triton import scaled_dot_product_attention as kg
from reference import sdpa_reference

g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16
q = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")
k = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")
v = (torch.randn(1, 1, 128, 64, generator=g) * 0.5).to(dt).to("npu:0")

out = kg(q, k, v, None, 0.0, False)
ref = sdpa_reference(q, k, v, None, 0.0, False, None, False).float()
d = (out.float() - ref).abs()

# 每个 M 块(64行)的 max err
for mb in range(2):
    rows = d[0, 0, mb * 64:(mb + 1) * 64]
    print(f"M块{mb}: max={rows.max().item():.4f} mean={rows.mean().item():.4f}")

# 假设 A: 输出等价于"只对前64个K做了attention"（第二块被丢弃）
refA = sdpa_reference(q, k, v, None, 0.0, False, None, False)
# 手工构造: mask 掉 K[64:]
import torch.nn.functional as Fn
m = torch.ones(1, 1, 128, 128, dtype=torch.bool, device="npu:0")
m[..., 64:] = False
refA = sdpa_reference(q, k, v, m, 0.0, False, None, False)
eA = (out.float() - refA.float()).abs().max().item()
print(f"假设A(只算前64 K): err={eA:.4f}")

# 假设 B: 第二块的贡献没被 alpha rescale（旧 acc 直接 += ）
# 数学: out = (A1 + A2) / (l1 + l2) 而 A_i 未乘各自 exp 偏移
qf, kf, vf = q.float(), k.float(), v.float()
s = qf @ kf.transpose(-1, -2) / 8.0
p = s.softmax(-1)
refB = p @ vf   # 正确参考本身
eB = (out.float() - refB).abs().max().item()
print(f"假设B(正确参考): err={eB:.4f}")

# 假设 C: V 的块偏移用错 stride（读错 V 行）→ 用 k 的行替换 v 检验不可行;
# 改为: out 是否匹配 attention(v=错位V)
v_shift = torch.roll(v, shifts=64, dims=2)
refC = sdpa_reference(q, k, v_shift, None, 0.0, False, None, False)
eC = (out.float() - refC.float()).abs().max().item()
print(f"假设C(V 循环位移64): err={eC:.4f}")

k_shift = torch.roll(k, shifts=64, dims=2)
refD = sdpa_reference(q, k_shift, v, None, 0.0, False, None, False)
eD = (out.float() - refD.float()).abs().max().item()
print(f"假设D(K 循环位移64): err={eD:.4f}")
