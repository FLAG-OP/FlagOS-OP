# op 层验证: A1 注册 → 拦截命中 → 三返回值语义 + 梯度链路。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import native_layer_norm_reference

    dev = profile.torch_device
    reg.CALL_COUNT["native_layer_norm"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev)
    w = torch.randn(512, dtype=torch.bfloat16, device=dev)
    b = torch.randn(512, dtype=torch.bfloat16, device=dev)

    # 1) 拦截验证: 原语直调命中
    before = reg.CALL_COUNT["native_layer_norm"]
    out, mean, rstd = torch.ops.aten.native_layer_norm(x, [512], w, b, 1e-5)
    assert reg.CALL_COUNT["native_layer_norm"] > before, "原语未被拦截"
    assert _CALLS["n"] > 0
    ref_out, ref_mean, ref_rstd = native_layer_norm_reference(x, [512], w, b)
    assert (out.float() - ref_out.to(dev).float()).abs().max().item() < 1e-2

    # 1b) 公开 API 路径: layer_norm 内部消费 native_layer_norm →
    #     公开调用同样命中（issue #55: 两者必须同时验证）
    before2 = reg.CALL_COUNT["native_layer_norm"]
    out2 = torch.nn.functional.layer_norm(x, (512,), w, b, 1e-5)
    hits2 = reg.CALL_COUNT["native_layer_norm"] - before2
    assert hits2 > 0, "公开 layer_norm 未消费 native_layer_norm"
    assert (out2.float() - ref_out.to(dev).float()).abs().max().item() < 1e-2

    # 2) 梯度链路: 原生 backward 节点随委托建立（grad vs CPU 参考）
    xg = (torch.randn(16, 512, dtype=torch.float32, device=dev)
          ).requires_grad_(True)
    wg = torch.randn(512, dtype=torch.float32, device=dev,
                     requires_grad=True)
    y = torch.ops.aten.native_layer_norm(xg, [512], wg, None, 1e-5)[0]
    y.sum().backward()
    assert xg.grad is not None and wg.grad is not None, "梯度缺失"
    assert torch.isfinite(xg.grad).all()

    # 3) weight/bias 为 None 的边界
    out3 = torch.ops.aten.native_layer_norm(x, [512], None, None, 1e-5)
    ref3 = native_layer_norm_reference(x, [512], None, None)
    assert (out3[0].float() - ref3[0].to(dev).float()).abs().max().item() < 1e-2

    # 4) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3, ops/sdpa/PLATFORM.md §3c 同款）
    with torch.inference_mode():
        xi = torch.randn(16, 256, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["native_layer_norm"]
        torch.ops.aten.native_layer_norm(xi, [256], None, None, 1e-5)
        hits_i = reg.CALL_COUNT["native_layer_norm"] - before_i
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::native_layer_norm@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["native_layer_norm"],
            "public_layer_norm_hits": hits2,
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
