# 数值调试: KernelGen 版误差结构定位
import sys

sys.path.insert(0, "/root/sdpatten-op")
sys.path.insert(0, "/root/sdpatten-op/intake_llm")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernelgen_triton import scaled_dot_product_attention as kg
from reference import sdpa_reference

g = torch.Generator(device="cpu").manual_seed(3)
dt = torch.float16

for S in (64, 128):  # 单块 / 双块
    q = (torch.randn(1, 1, S, 64, generator=g) * 0.5).to(dt).to("npu:0")
    k = (torch.randn(1, 1, S, 64, generator=g) * 0.5).to(dt).to("npu:0")
    v = (torch.randn(1, 1, S, 64, generator=g) * 0.5).to(dt).to("npu:0")
    out = kg(q, k, v, None, 0.0, False)
    ref = sdpa_reference(q, k, v, None, 0.0, False, None, False).float()
    d = (out.float() - ref).abs()
    print(f"S={S} noncausal: max={d.max().item():.4f} "
          f"mean={d.mean().item():.4f}")
    # 行级: 前半 vs 后半
    print(f"  行0 err={d[0,0,0].max().item():.4f}  "
          f"行S-1 err={d[0,0,-1].max().item():.4f}")
    # 归一性检查: out 行和应≈V 行和均值加权; 检查 out 是否像未归一化
    print(f"  out[0,0,0,:4] = {out[0,0,0,:4].float().tolist()}")
    print(f"  ref[0,0,0,:4] = {ref[0,0,0,:4].tolist()}")

# 猜想验证: 如果是 l_i 归一化错(如 p 没除以 l), 输出幅度会偏大
# 检查 out/ref 的行均值比
q = (torch.randn(1, 1, 64, 64, generator=g) * 0.5).to(dt).to("npu:0")
k = (torch.randn(1, 1, 64, 64, generator=g) * 0.5).to(dt).to("npu:0")
v = (torch.randn(1, 1, 64, 64, generator=g) * 0.5).to(dt).to("npu:0")
out = kg(q, k, v, None, 0.0, False)
ref = sdpa_reference(q, k, v, None, 0.0, False, None, False).float()
ratio = out.float().abs().mean() / ref.abs().mean()
print(f"\n幅度比 out/ref = {ratio.item():.4f}（≈1 正常; 偏大→未归一化）")
# V 行均值 vs out 行均值: softmax 均匀时 out≈V 均值
print(f"V mean={v.float().mean().item():.4f}  out mean="
      f"{out.float().mean().item():.4f}  ref mean={ref.mean().item():.4f}")
