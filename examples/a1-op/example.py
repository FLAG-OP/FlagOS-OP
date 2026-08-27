#!/usr/bin/env python3
"""样例 A1×算子层: 从零实现 Triton relu 并接管 torch.relu。

完全自包含（不依赖 routes/），演示 aten 路线四步:
  ① 写 Triton kernel
  ② 注册进 aten（dispatch key 来自设备 profile）
  ③ 拦截验证（torch 真实调用命中我们的 kernel）
  ④ 精度 + 性能验证

运行: python3 examples/a1-op/example.py [设备profile名]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ============ ① Triton kernel ============
import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def relu_kernel(x):
    # pointwise_dynamic 自动处理 shape 广播/block 划分/多 dtype
    return tl.maximum(x, 0.0)


def relu_triton(x: torch.Tensor) -> torch.Tensor:
    return relu_kernel(x)


# ============ ② aten 注册 ============
_LIB = None
CALLS = {"relu": 0}


def relu_counter(self):
    """aten::relu 签名: relu(Tensor self) -> Tensor"""
    CALLS["relu"] += 1
    return relu_triton(self)


def register_relu(dispatch_key: str):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("relu", relu_counter, dispatch_key)


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    dev = profile.torch_device
    print(f"[a1-op 样例] 设备: {profile.summary()}")

    # ============ ③ 拦截验证 ============
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
    register_relu(profile.dispatch_key)
    out = torch.relu(x)  # 任意框架代码都会命中
    assert CALLS["relu"] == 1, "未被拦截"
    print(f"  拦截成功: torch.relu -> Triton kernel (calls={CALLS['relu']})")

    # ============ ④ 精度 ============
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        xx = torch.randn(4096, 4096, dtype=dt, device=dev)
        err = (torch.relu(xx).float()
               - torch.relu(xx.cpu()).to(dev).float()).abs().max().item()
        assert err == 0.0, f"{dt} 精度错误 {err}"
    print("  精度: bf16/fp16/fp32 位级一致（relu 为精确算子）")

    # ============ ④ 性能（快速采样） ============
    # 注: 共享环境下绝对值波动大（本机同 kernel 曾观测 0.04~16ms 差异），
    # 以干净进程微基准为准；此处仅验证可用性
    def bench(fn, iters=100, warm=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_tri = bench(lambda: relu_triton(x))
    t_ref = bench(lambda: torch.relu(x.cpu()), iters=20)
    print(f"  性能(采样): Triton={t_tri:.3f}ms  CPU={t_ref:.3f}ms  "
          f"加速={t_ref/t_tri:.1f}x（绝对值视环境负载而定）")
    print("=> A1×op 样例 PASS")


if __name__ == "__main__":
    main()
