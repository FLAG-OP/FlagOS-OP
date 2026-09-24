# op 层验证: result_type 不可 A1 注册（自用/实验），验证 torch 级一致。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from kernel.torch_level import result_type_torch

    dev = profile.torch_device
    try:
        reg.register_a1(profile.dispatch_key)
        raise AssertionError("不应被注册")
    except NotImplementedError:
        pass
    for a_name, b_name in [("float16", "float32"), ("int32", "float32"),
                           ("bfloat16", "int64"), ("float32", "float64")]:
        a = torch.empty(2, dtype=getattr(torch, a_name), device=dev)
        b = torch.empty(2, dtype=getattr(torch, b_name), device=dev)
        assert result_type_torch(a, b) == torch.result_type(a, b)
    return {"ok": True, "registered": None, "route": reg.ROUTE,
            "intercepted_calls": 0}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
