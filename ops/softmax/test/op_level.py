# op 层验证: A1 注册 → 拦截命中 → 精度不变形 + dtype 语义 + 回退正确。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import softmax_reference

    dev = profile.torch_device
    reg.CALL_COUNT["softmax"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev) * 8

    # 1) 拦截验证: aten 直调（dim 位置 + dtype 缺省）命中
    before = reg.CALL_COUNT["softmax"]
    out = torch.ops.aten.softmax(x, -1)
    assert reg.CALL_COUNT["softmax"] > before, "aten::softmax 未被拦截"
    assert _CALLS["n"] > 0
    assert (out.float() - softmax_reference(x, -1).to(dev).float()
            ).abs().max().item() < 1e-2

    # 1b) 公开 API 三路径同样命中（F.softmax / Tensor.softmax / torch.softmax）
    for name, fn in [("F.softmax", lambda t: torch.nn.functional.softmax(t, -1)),
                     ("Tensor.softmax", lambda t: t.softmax(-1)),
                     ("torch.softmax", lambda t: torch.softmax(t, -1))]:
        before2 = reg.CALL_COUNT["softmax"]
        fn(x)
        assert reg.CALL_COUNT["softmax"] > before2, f"{name} 未命中"

    # 2) dtype 参数语义: fp16 → fp32 输出（half_to_float 等价路径）
    xh = torch.randn(32, 512, dtype=torch.float16, device=dev) * 8
    got = torch.ops.aten.softmax(xh, -1, dtype=torch.float32)
    assert got.dtype == torch.float32
    assert (got - softmax_reference(xh, -1, torch.float32).to(dev)
            ).abs().max().item() < 2e-5

    # 3) 非尾维 dim（中间维）经注册路径
    x3 = torch.randn(4, 64, 128, dtype=torch.float16, device=dev)
    got3 = torch.ops.aten.softmax(x3, 1)
    exp3 = softmax_reference(x3, 1).to(dev)
    assert (got3.float() - exp3.float()).abs().max().item() < 1e-2

    # 4) 梯度链路（原生 SoftmaxBackward）
    xg = (torch.randn(32, 256, dtype=torch.float32, device=dev) * 4
          ).requires_grad_(True)
    torch.ops.aten.softmax(xg, -1).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    # softmax 梯度性质: sum(y * dy) 行扣除 → y(1-y)·g 校验
    y = torch.softmax(xg.detach(), -1)
    g = torch.randn_like(y)
    (torch.ops.aten.softmax(xg, -1) * g).sum().backward()
    ref_g = y * (g - (y * g).sum(-1, keepdim=True))
    assert (xg.grad - ref_g).abs().max().item() < 1e-4, "softmax 梯度公式"

    # 5) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        xi = torch.randn(16, 256, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["softmax"]
        torch.ops.aten.softmax(xi, -1)
        hits_i = reg.CALL_COUNT["softmax"] - before_i
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::softmax@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["softmax"],
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
