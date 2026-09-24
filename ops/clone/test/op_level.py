# op 层验证: A1 aten 注册 → 拦截命中 → 精度/存储独立不变形。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import clone_reference

    dev = profile.torch_device

    # 注册前先取原生参考（注册后 .clone() 会命中我们的实现）
    x = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
    exp = clone_reference(x)
    xi = torch.randint(-100, 100, (64, 64), dtype=torch.int32, device=dev)
    xi_ref = clone_reference(xi)

    reg.CALL_COUNT["clone"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    # 1) 拦截验证
    before = reg.CALL_COUNT["clone"]
    out = x.clone()
    assert reg.CALL_COUNT["clone"] > before, "aten::clone 未被拦截"
    assert torch.equal(out, exp), "拦截后数值不一致"
    assert out.data_ptr() != x.data_ptr(), "拦截后仍共享存储"

    # 2) memory_format 透传（contiguous_format）
    y = torch.randn(32, 64, device=dev).t()
    yc = y.clone(memory_format=torch.contiguous_format)
    assert yc.is_contiguous(), "contiguous_format 未生效"

    # 3) 多 dtype + 存储独立
    for dt in (torch.float16, torch.float32, torch.bfloat16):
        a = torch.randn(256, 128, dtype=dt, device=dev)
        b = a.clone()
        assert b.dtype == dt and b.data_ptr() != a.data_ptr()
        assert torch.equal(b, a)

    # 4) 回退: 不支持 dtype（int32）仍正确
    assert torch.equal(xi.clone(), xi_ref), "int32 回退不一致"

    return {"ok": True,
            "registered": f"aten::clone@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["clone"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
