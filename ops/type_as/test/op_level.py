# op 层验证: A1 aten 注册 → 拦截命中 → 精度不变形 + 回退正确。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))                 # 引用同算子的 register/kernel
sys.path.insert(0, str(OP_DIR.parent.parent))   # 仓库根（common 等）


def run(profile):
    import torch

    import register as reg
    from reference import type_as_reference

    dev = profile.torch_device
    reg.CALL_COUNT["type_as"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    # 1) 拦截验证: torch 宿主调用必须命中我们的实现
    x = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
    other = torch.empty(4, dtype=torch.float32, device=dev)
    before = reg.CALL_COUNT["type_as"]
    out = torch.ops.aten.type_as.default(x, other)
    after = reg.CALL_COUNT["type_as"]
    assert after > before, "aten::type_as 未被拦截"
    assert torch.equal(out, type_as_reference(x, other)), "拦截后精度不一致"

    # 1b) 方法式调用同样命中
    x2 = torch.randn(64, 128, dtype=torch.float16, device=dev)
    o2 = torch.empty(4, dtype=torch.bfloat16, device=dev)
    out2 = x2.type_as(o2)
    assert reg.CALL_COUNT["type_as"] > after, "Tensor.type_as 未命中"
    assert torch.equal(out2, type_as_reference(x2, o2))

    # 2) 精度: 多 dtype 组合经注册路径不改变语义
    for sdt, tdt in [(torch.float32, torch.float16),
                     (torch.float16, torch.float32),
                     (torch.bfloat16, torch.float32),
                     (torch.float32, torch.bfloat16)]:
        a = torch.randn(1024, 512, dtype=sdt, device=dev) * 2
        b = torch.empty(4, dtype=tdt, device=dev)
        got = torch.ops.aten.type_as.default(a, b)
        exp = type_as_reference(a, b)
        assert got.dtype == tdt and torch.equal(got, exp), f"{sdt}->{tdt}"

    # 3) 别名与回退: 同 dtype 返回 self；不支持 dtype 走原生回退
    y = torch.randn(64, 64, dtype=torch.float16, device=dev)
    assert torch.ops.aten.type_as.default(y, y) is y, "same-dtype 未别名"
    xi = torch.randint(-500, 500, (128, 128), dtype=torch.int32, device=dev)
    oi = torch.empty(4, dtype=torch.float32, device=dev)
    assert torch.equal(torch.ops.aten.type_as.default(xi, oi),
                       type_as_reference(xi, oi)), "int32 回退不一致"

    return {"ok": True,
            "registered": f"aten::type_as@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["type_as"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
