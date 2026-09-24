# op 层验证: detach_ 不可 A1 注册（route=自用/实验），验证 torch 级语义。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from kernel.torch_level import detach__torch

    dev = profile.torch_device
    try:
        reg.register_a1(profile.dispatch_key)
        raise AssertionError("不应被注册")
    except NotImplementedError:
        pass
    x = torch.randn(128, 256, dtype=torch.float32, device=dev)
    x.requires_grad_(True)
    got = detach__torch(x)
    assert got.data_ptr() == x.data_ptr() and got.requires_grad is False
    return {"ok": True, "registered": None, "route": reg.ROUTE,
            "intercepted_calls": 0}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
