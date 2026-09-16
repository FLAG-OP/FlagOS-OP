# extreme case fp32 误差解剖: 绝对 vs 相对，以及与原生实现同口径对比
import sys

sys.path.insert(0, "/root/sdpatten-op")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernel.triton_level import sdpa_triton  # noqa: E402

# 复现 extreme case（gen_golden extreme: B1 H4 S128 D64 fp32 causal scale8）
seed = None
import json  # noqa: E402
from pathlib import Path  # noqa: E402

idx = json.loads(Path("/root/sdpatten-op/goldendata/index.json").read_text())
ext = [f for f in idx["files"]
       if "extreme_float32" in f["file"] and "c1_mnone" in f["file"]][0]
d = torch.load("/root/sdpatten-op/goldendata/" + ext["file"])
q, k, v = (t.to("npu:0") for t in d["inputs"][:3])
ref = d["expected"].to("npu:0")

out = sdpa_triton(q, k, v, None, 0.0, True, None, False)
diff = (out - ref).abs()
rel = diff / ref.abs().clamp_min(1e-6)
print(f"abs max: {diff.max().item():.3e}")
print(f"ref 幅值范围: [{ref.abs().min().item():.3e}, "
      f"{ref.abs().max().item():.3e}]")
print(f"rel max (逐元素): {rel.max().item():.3e}")

# 原生 fp32 同输入（独立参照）
native = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True)
nd = (native - ref).abs()
print(f"native abs max vs golden: {nd.max().item():.3e}")
print(f"ours  vs native abs max:  "
      f"{(out - native).abs().max().item():.3e}")

# softmax 饱和度: P 的行最大值（=1 表示完全 one-hot）
qf, kf, vf = q.float(), k.float(), v.float()
s = torch.matmul(qf, kf.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
p = torch.softmax(s.masked_fill(
    ~torch.tril(torch.ones(128, 128, dtype=torch.bool, device="npu:0")),
    float("-inf")), dim=-1)
print(f"P 行最大值 mean={p.max(dim=-1).values.mean().item():.4f} "
      f"(≈1 → 饱和 one-hot)")
