# 守卫行为验证: ①npu 正常 ②cpu tensor 触发调用守卫 ③元数据可读
import sys

sys.path.insert(0, "/root/sdpatten-op")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernel.triton_level import (PLATFORM, SUPPORTED_DEVICE_TYPES,  # noqa: E402
                                 sdpa_triton)

assert PLATFORM == "ascend910" and SUPPORTED_DEVICE_TYPES == ("npu",)
print(f"元数据 OK: PLATFORM={PLATFORM} types={SUPPORTED_DEVICE_TYPES}")

# ① npu 正常路径
g = torch.Generator(device="cpu").manual_seed(1)
q = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16).to("npu:0")
k = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16).to("npu:0")
v = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16).to("npu:0")
out = sdpa_triton(q, k, v, None, 0.0, True, None, False)
print("npu 路径 OK:", tuple(out.shape))

# ② cpu tensor 触发调用守卫
qc, kc, vc = q.cpu(), k.cpu(), v.cpu()
try:
    sdpa_triton(qc, kc, vc, None, 0.0, True, None, False)
    print("FAIL: cpu 调用未被拦截!")
    sys.exit(1)
except RuntimeError as e:
    assert "ascend910" in str(e) and "PLATFORM.md" in str(e)
    print("调用守卫 OK:", str(e)[:60], "...")

# ③ 注册守卫（模拟: 无 torch_npu 时不该注册——本机有 npu，验证正向可注册）
from register import register_a1
lib = register_a1("AutogradPrivateUse1")
print("注册守卫(正向) OK: npu 环境可注册")
del lib
print("ALL GUARD CHECKS PASS")
