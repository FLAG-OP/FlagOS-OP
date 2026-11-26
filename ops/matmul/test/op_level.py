# op 层验证: A1 注册 → 拦截命中 → 精度不变形（GEMM rel）+ 梯度链路。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import matmul_reference

    dev = profile.torch_device
    reg.CALL_COUNT["matmul"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    a = torch.randn(64, 128, dtype=torch.bfloat16, device=dev)
    b = torch.randn(128, 96, dtype=torch.bfloat16, device=dev)

    # 1) 拦截验证: aten 直调 + torch.matmul + @ 运算符
    before = reg.CALL_COUNT["matmul"]
    out = torch.ops.aten.matmul(a, b)
    assert reg.CALL_COUNT["matmul"] > before, "aten::matmul 未被拦截"
    assert _CALLS["n"] > 0
    exp = matmul_reference(a, b).to(dev)
    rel = ((out.float() - exp.float()).abs()
           / exp.float().abs().clamp_min(1.0)).max().item()
    assert rel < 5e-2, f"拦截后精度 rel={rel}"

    before2 = reg.CALL_COUNT["matmul"]
    _ = torch.matmul(a, b)
    assert reg.CALL_COUNT["matmul"] > before2, "torch.matmul 未命中"
    before3 = reg.CALL_COUNT["matmul"]
    _ = a @ b
    assert reg.CALL_COUNT["matmul"] > before3, "@ 运算符未命中"

    # 2) 梯度链路（原生 mm backward 随委托建立）
    ag = torch.randn(16, 32, dtype=torch.float32, device=dev,
                     requires_grad=True)
    bg = torch.randn(32, 16, dtype=torch.float32, device=dev,
                     requires_grad=True)
    torch.ops.aten.matmul(ag, bg).sum().backward()
    assert ag.grad is not None and torch.isfinite(ag.grad).all()

    # 3) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        ai = torch.randn(16, 32, dtype=torch.float16, device=dev)
        bi = torch.randn(32, 16, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["matmul"]
        torch.ops.aten.matmul(ai, bi)
        hits_i = reg.CALL_COUNT["matmul"] - before_i
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::matmul@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["matmul"],
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
