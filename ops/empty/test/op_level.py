# op 层验证: empty 属分配原语，**无 A1 注册**（route=自用/实验）。
# 验证: register_a1 按设计拒绝；torch 级元数据与原生一致。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from kernel.torch_level import empty_torch
    from reference import empty_reference

    dev = profile.torch_device

    try:
        reg.register_a1(profile.dispatch_key)
        raise AssertionError("empty 不应被注册")
    except NotImplementedError:
        pass

    for shape, dt in [((128, 256), torch.float32),
                      ((8, 5000), torch.bfloat16),
                      ((2, 3, 16, 16), torch.float16)]:
        got = empty_torch(shape, dtype=dt, device=dev)
        exp = empty_reference(shape, dtype=dt)
        assert tuple(got.shape) == shape and got.dtype == dt
        assert got.stride() == exp.stride()

    return {"ok": True, "registered": None, "route": reg.ROUTE,
            "intercepted_calls": 0}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
