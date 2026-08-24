#!/usr/bin/env python3
"""A1 算子层测试: aten 注册 → torch 真实调用被拦截 → 精度对齐 CPU 参考"""
from __future__ import annotations


def run(profile) -> bool:
    import torch
    from routes.a1_aten import register_aten

    dev = profile.torch_device
    print("=" * 60)
    print(f"A1 op-level [{profile.name}]: Triton -> aten -> F.gelu @ {dev}")
    print("=" * 60)

    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev) * 2
    ref = torch.nn.functional.gelu(x.cpu(), approximate="tanh").to(dev)

    lib = register_aten.register_gelu_aten(profile.dispatch_key)
    assert register_aten.CALL_COUNT["gelu"] == 0

    out = torch.nn.functional.gelu(x, approximate="tanh")
    assert register_aten.CALL_COUNT["gelu"] == 1, "kernel 未被拦截!"

    err = (out.float() - ref.float()).abs().max().item()
    print(f"  intercepted calls : {register_aten.CALL_COUNT['gelu']}")
    print(f"  max_err vs CPU ref: {err:.3e}")
    assert err < 1e-1, f"精度超差: {err}"

    for dt in (torch.float16, torch.float32):
        xx = torch.randn(2048, 2048, dtype=dt, device=dev)
        rr = torch.nn.functional.gelu(xx.cpu(), approximate="tanh").to(dev)
        oo = torch.nn.functional.gelu(xx, approximate="tanh")
        assert (oo.float() - rr.float()).abs().max().item() < 1e-1
    print("  dtype sweep       : PASS (fp16/fp32)")
    print("  => A1 op-level PASS")
    del lib
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    name = sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"
    run(load_profile(name))
