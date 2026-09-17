# 最小冒烟: sdpa_triton 单 case 编译 + 数值
import sys
from pathlib import Path

sys.path.insert(0, "/root/sdpatten-op")

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernel.triton_level import sdpa_triton  # noqa: E402
from reference import sdpa_reference  # noqa: E402

g = torch.Generator(device="cpu").manual_seed(42)
dt = torch.float16
dev = "npu:0"
q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)

out = sdpa_triton(q, k, v, None, 0.0, True, None, False)
ref = sdpa_reference(q, k, v, None, 0.0, True, None, False).float()
err = (out.float() - ref).abs().max().item()
print(f"causal fp16 basic: err={err:.3e}  shape={tuple(out.shape)}")
assert err < 2e-2, "超容差"

# 尾块 case
q2 = (torch.randn(1, 4, 100, 64, generator=g) * 0.5).to(dt).to(dev)
k2 = (torch.randn(1, 4, 100, 64, generator=g) * 0.5).to(dt).to(dev)
v2 = (torch.randn(1, 4, 100, 64, generator=g) * 0.5).to(dt).to(dev)
out2 = sdpa_triton(q2, k2, v2, None, 0.0, False, None, False)
ref2 = sdpa_reference(q2, k2, v2, None, 0.0, False, None, False).float()
err2 = (out2.float() - ref2).abs().max().item()
print(f"tail100 fp16 noncausal: err={err2:.3e}")
assert err2 < 2e-2, "尾块超容差"
print("SMOKE OK")
