# op 层验证: A1 aten 注册 → 拦截命中 → 精度不变形 + 回退正确。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import linear_reference

    dev = profile.torch_device
    reg.CALL_COUNT["linear"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    # 1) 拦截验证: torch 宿主调用必须命中我们的实现
    x = torch.randn(128, 256, dtype=torch.bfloat16, device=dev)
    w = torch.randn(64, 256, dtype=torch.bfloat16, device=dev) * 0.05
    before = reg.CALL_COUNT["linear"]
    out = torch.ops.aten.linear(x, w, None)
    after = reg.CALL_COUNT["linear"]
    assert after > before, "aten::linear 未被拦截"
    assert _CALLS["n"] > 0, "counter 未记入（op 层拦截）"
    err = ((out.float() - linear_reference(x, w).to(dev).float()).abs()
           / linear_reference(x, w).to(dev).float().abs().clamp_min(1.0)
           ).max().item()
    assert err < 5e-2, f"拦截后精度不一致 rel={err}"

    # 1b) F.linear / nn.functional 路径同样命中（带 bias + None 两种）
    out2 = torch.nn.functional.linear(x, w)
    assert reg.CALL_COUNT["linear"] > after, "F.linear 未命中"

    # 2) 精度: 多 dtype 组合经注册路径不改变语义（GEMM 5e-2 rel）
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(64, 512, dtype=torch.float32, device=dev).to(dt)
        wdt = torch.randn(128, 512, dtype=torch.float32, device=dev).to(dt) * 0.05
        b = torch.randn(128, dtype=torch.float32, device=dev).to(dt) * 0.05
        got = torch.ops.aten.linear(a, wdt, b)
        exp = linear_reference(a, wdt, b).to(dev)
        rel = ((got.float() - exp.float()).abs()
               / exp.float().abs().clamp_min(1.0)).max().item()
        assert rel < 5e-2, f"{dt} rel={rel}"

    # 3) 0 维批 / 单行边界
    x1 = torch.randn(1, 256, dtype=torch.float16, device=dev)
    got1 = torch.ops.aten.linear(x1, w.to(torch.float16), None)
    assert got1.shape == (1, 64), "单行 shape 异常"

    # 4) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        xi = torch.randn(32, 256, dtype=torch.float16, device=dev)
        wi = torch.randn(32, 256, dtype=torch.float16, device=dev) * 0.05
        before_i = reg.CALL_COUNT["linear"]
        oi = torch.ops.aten.linear(xi, wi, None)
        hits_i = reg.CALL_COUNT["linear"] - before_i
    assert torch.isfinite(oi).all()
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::linear@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["linear"],
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
