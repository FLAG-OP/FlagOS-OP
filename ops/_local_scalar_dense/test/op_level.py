# op 层验证: A1 注册 → 拦截命中 → host 标量值正确。
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
    for dt in (torch.float32, torch.float16, torch.int32, torch.bool):
        val = True if dt is torch.bool else (1.5 if dt.is_floating_point else 9)
        x = torch.full((1,), val, dtype=dt, device=dev)
        exp = x.item()

    reg.CALL_COUNT["_local_scalar_dense"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None
    before = reg.CALL_COUNT["_local_scalar_dense"]
    x = torch.full((1,), 3.5, dtype=torch.float32, device=dev)
    got = torch.ops.aten._local_scalar_dense.default(x)
    assert reg.CALL_COUNT["_local_scalar_dense"] > before, "未拦截"
    assert got == 3.5, got
    return {"ok": True,
            "registered": "aten::_local_scalar_dense@" + profile.dispatch_key,
            "intercepted_calls": reg.CALL_COUNT["_local_scalar_dense"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
