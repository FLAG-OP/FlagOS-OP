#!/usr/bin/env python3
"""跨层一致性验证（L0 kernel 直调 ↔ L2 dispatch 调用）。

锚定算子 gelu_and_mul，同一输入张量在四处输出两两比对:
  L0-A: A2 Triton kernel 直调
  L0-C: 自研 C++ kernel 直调（JIT 编译产物）
  L2:   call_op("gelu_and_mul") 经 dispatch 选择 flagos 实现
  ref:  PyTorch 语义参考
"""
from __future__ import annotations


def run(profile) -> bool:
    import os

    import torch
    from routes.a2_dispatch.plugin import kernels as K

    dev = profile.torch_device
    os.environ["VLLM_FL_PLUGIN_MODULES"] = \
        "routes.a2_dispatch.plugin.register_ops"

    print("=" * 60)
    print(f"Consistency L0<->L2 [{profile.name}] @ {dev}")
    print("=" * 60)

    # 统一输入（同 seed 同张量）
    torch.manual_seed(20260824)
    shape, dt = (512, 2048), torch.bfloat16
    x = torch.randn(*shape, dtype=dt, device=dev) * 2
    y = torch.randn(*shape, dtype=dt, device=dev)

    ref = K.gelu_and_mul_reference(x, y)

    # L0-A: Triton 直调
    out_triton = K.gelu_and_mul_triton(x, y)

    # L0-C: 自研 C++ kernel（JIT 失败则跳过该实现）
    out_csrc = None
    try:
        from tests.kernel_level.test_b import _load_csrc_module
        mod = _load_csrc_module()
        out_csrc = mod.gelu_and_mul(x, y)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  自研 C++ kernel: SKIP（{str(e)[:50]}）")

    # L2: dispatch 调用（flagos 优先）
    from vllm_fl.dispatch import get_default_manager, call_op
    from vllm_fl.dispatch.policy import with_preference

    m = get_default_manager()
    m.ensure_initialized()
    with with_preference("flagos"):
        out_dispatch = call_op("gelu_and_mul", None, x, y)
    used = m._called_ops["gelu_and_mul"]
    print(f"  L2 选中实现: {used}")
    assert used == "default.flagos"

    # ---- 两两比对 ----
    impls = {"triton_L0": out_triton, "dispatch_L2": out_dispatch,
             "reference": ref}
    if out_csrc is not None:
        impls["csrc_L0"] = out_csrc

    names = list(impls)
    tol = 1e-2
    print(f"\n  {'':14s}" + "".join(f"{n:>14s}" for n in names))
    all_ok = True
    for a in names:
        row = f"  {a:14s}"
        for b in names:
            if a == b:
                row += f"{'—':>14s}"
                continue
            err = (impls[a].float() - impls[b].float()).abs().max().item()
            ok = err < tol
            all_ok &= ok
            row += f"{err:>14.2e}"
        print(row)

    print(f"\n  容差: {tol}（bf16 跨实现）")
    assert all_ok, "跨层一致矩阵存在超差项"
    print("  => Consistency L0<->L2 PASS")
    return True
