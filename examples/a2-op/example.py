#!/usr/bin/env python3
"""样例 A2×算子层: Triton 融合算子 → FlagOS dispatch 插件 → 三段式。

自包含演示:
  ① Triton kernel + PyTorch 参考实现
  ② 插件式注册双后端（default.flagos 150 + reference.torch 50）
  ③ 三段式: 精度 / dispatch 策略切换 / 性能

运行: python3 examples/a2-op/example.py [设备profile名]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


# ============ ① Triton kernel + 参考 ============
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_and_mul_kernel(x, y):
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    tanh_in = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    return 0.5 * xf * (1.0 + tanh_in) * y


def impl_triton(x, gate):
    return gelu_and_mul_kernel(x, gate)


def impl_reference(x, gate):
    xf = x.float()
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + torch.tanh(inner)) * gate.float()).to(x.dtype)


# ============ 性能回归用例（scripts/perf_run.py 消费） ============
def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            x = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device) * 2
            y = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device)
            return lambda: fn(x, y)
        return _make

    def bw_3t(t):  # 读 x,y 写 out → 3 张量 bf16
        return {"GBps": 8192 * 8192 * 2 * 3 / t / 1e6}

    return [
        PerfCase("example.a2-op.gelu_and_mul.triton", group="example",
                 level="op", make_fn=make(impl_triton), derived=bw_3t),
        PerfCase("example.a2-op.gelu_and_mul.reference", group="example",
                 level="op", make_fn=make(impl_reference)),
    ]


# ============ ② 插件注册（内联演示，与 routes/a2_dispatch 同模式） ============
def register_builtins(registry) -> None:
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
    from vllm_fl.dispatch.backends.base import Backend

    class TritonBackend(Backend):
        @property
        def name(self): return "flagos"

        def is_available(self):
            try:
                import triton, flag_gems  # noqa
                return True
            except ImportError:
                return False

        def op(self, obj, x, gate):
            return impl_triton(x, gate)

    class RefBackend(Backend):
        @property
        def name(self): return "reference"

        def is_available(self): return True

        def op(self, obj, x, gate):
            return impl_reference(x, gate)

    fl, ref = TritonBackend(), RefBackend()
    registry.register_many([
        OpImpl(op_name="gelu_and_mul", impl_id="default.flagos",
               kind=BackendImplKind.DEFAULT, fn=fl.op, vendor=None,
               priority=BackendPriority.DEFAULT),
        OpImpl(op_name="gelu_and_mul", impl_id="reference.torch",
               kind=BackendImplKind.REFERENCE, fn=ref.op, vendor=None,
               priority=BackendPriority.REFERENCE),
    ])


# ⚠️ discovery 实际查找 register / vllm_fl_register
register = register_builtins
vllm_fl_register = register_builtins


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    dev = profile.torch_device
    os.environ["VLLM_FL_PLUGIN_MODULES"] = __name__  # 本模块即插件
    print(f"[a2-op 样例] 设备: {profile.summary()}")

    # ---- 三段式 1/3: 精度 ----
    for shape in [(4096, 4096), (1, 14336), (128, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            y = torch.randn(*shape, dtype=dt, device=dev)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            assert (impl_triton(x, y).float()
                    - impl_reference(x, y).float()).abs().max().item() < tol
    print("  精度: 9/9 组合 PASS")

    # ---- 三段式 2/3: dispatch ----
    from vllm_fl.dispatch import get_default_manager, call_op
    from vllm_fl.dispatch.policy import with_preference

    m = get_default_manager()
    m.ensure_initialized()
    x = torch.randn(512, 512, dtype=torch.bfloat16, device=dev) * 2
    y = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
    with with_preference("flagos"):
        call_op("gelu_and_mul", None, x, y)
        assert m._called_ops["gelu_and_mul"] == "default.flagos"
    with with_preference("reference"):
        call_op("gelu_and_mul", None, x, y)
        assert m._called_ops["gelu_and_mul"] == "reference.torch"
    print("  dispatch: flagos→default.flagos / reference→reference.torch PASS")

    # ---- 三段式 3/3: 性能（短采样，避免分配器池增长失真） ----
    shape = (8192, 8192)
    x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
    y = torch.randn(*shape, dtype=torch.bfloat16, device=dev)

    def bench(fn, iters=100, warm=20):
        for _ in range(warm):
            fn(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x, y)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_tri, t_ref = bench(impl_triton), bench(impl_reference)
    bw = x.numel() * 2 * 3 / t_tri / 1e9 * 1e3
    print(f"  性能: Triton={t_tri:.3f}ms ({bw:.0f} GB/s)  "
          f"参考={t_ref:.1f}ms  加速={t_ref/t_tri:.0f}x")
    print("=> A2×op 样例 PASS")


if __name__ == "__main__":
    main()
