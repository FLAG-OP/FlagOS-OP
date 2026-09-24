# op 层验证: A1 注册 → 拦截命中 → 别名/requires_grad 不变形。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg

    dev = profile.torch_device
    x = torch.randn(256, 128, dtype=torch.float32, device=dev)
    x.requires_grad_(True)
    exp_ptr, exp_stride = x.data_ptr(), tuple(x.stride())

    reg.CALL_COUNT["detach"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None
    before = reg.CALL_COUNT["detach"]
    got = torch.ops.aten.detach.default(x)
    assert reg.CALL_COUNT["detach"] > before, "未拦截"
    assert got.data_ptr() == exp_ptr and got.stride() == exp_stride
    assert got.requires_grad is False
    return {"ok": True,
            "registered": "aten::detach@" + profile.dispatch_key,
            "intercepted_calls": reg.CALL_COUNT["detach"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
