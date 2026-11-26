# op 层验证: A1 注册 → 拦截命中 → 精度不变形 + native 原语配对验证。
#
# issue #55: layer_norm 与 native_layer_norm 必须同时验证——本测试在
# 注册 layer_norm 后验证公开路径拦截; native_layer_norm 原语的独立
# 拦截由 ops/native_layer_norm/test/op_level.py 验证（两者消费关系:
# 公开 API 的注册实现委托原生 layer_norm, 不再自然落到原语注册面）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import layer_norm_reference

    dev = profile.torch_device
    reg.CALL_COUNT["layer_norm"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev)
    w = torch.randn(512, dtype=torch.bfloat16, device=dev)
    b = torch.randn(512, dtype=torch.bfloat16, device=dev)

    # 1) 拦截验证: aten 直调 + F.layer_norm + nn.Module 三条路径
    before = reg.CALL_COUNT["layer_norm"]
    out = torch.ops.aten.layer_norm(x, [512], w, b, 1e-5, False)
    assert reg.CALL_COUNT["layer_norm"] > before, "aten::layer_norm 未被拦截"
    assert _CALLS["n"] > 0
    exp = layer_norm_reference(x, [512], w, b).to(dev)
    assert (out.float() - exp.float()).abs().max().item() < 1e-2

    out2 = torch.nn.functional.layer_norm(x, (512,), w, b, 1e-5)
    assert reg.CALL_COUNT["layer_norm"] > before + 1, "F.layer_norm 未命中"
    assert (out2.float() - exp.float()).abs().max().item() < 1e-2

    # 2) 精度: 三 dtype 组合
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(32, 1024, dtype=torch.float32, device=dev).to(dt)
        wd = torch.randn(1024, dtype=torch.float32, device=dev).to(dt)
        got = torch.ops.aten.layer_norm(a, [1024], wd, wd, 1e-5, False)
        exp_d = layer_norm_reference(a, [1024], wd, wd).to(dev)
        tol = 2e-5 if dt is torch.float32 else 1e-2
        err = (got.float() - exp_d.float()).abs().max().item()
        assert err < tol, f"{dt} err={err}"

    # 3) None weight/bias 边界
    out3 = torch.ops.aten.layer_norm(x, [512], None, None, 1e-5, False)
    exp3 = layer_norm_reference(x, [512], None, None).to(dev)
    assert (out3.float() - exp3.float()).abs().max().item() < 1e-2

    # 4) 梯度链路（原生 LayerNormBackward）
    xg = torch.randn(16, 512, dtype=torch.float32, device=dev,
                     requires_grad=True)
    torch.ops.aten.layer_norm(xg, [512], None, None, 1e-5, False) \
        .sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()

    # 5) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        xi = torch.randn(16, 256, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["layer_norm"]
        torch.ops.aten.layer_norm(xi, [256], None, None, 1e-5, False)
        hits_i = reg.CALL_COUNT["layer_norm"] - before_i
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::layer_norm@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["layer_norm"],
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
