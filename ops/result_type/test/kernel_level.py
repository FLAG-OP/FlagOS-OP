# kernel 层直测（dtype 元数据类）: 类型提升矩阵。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

_DTS = ["bfloat16", "float16", "float32", "float64",
        "int8", "int32", "int64", "bool"]


def run(profile):
    import torch

    from kernel.torch_level import result_type_torch

    dev = profile.torch_device
    checks = 0
    for a_name in _DTS:
        for b_name in _DTS:
            a = torch.empty(2, dtype=getattr(torch, a_name), device=dev)
            b = torch.empty(2, dtype=getattr(torch, b_name), device=dev)
            got = result_type_torch(a, b)
            exp = torch.result_type(a, b)
            assert got == exp, f"{a_name},{b_name}: {got} != {exp}"
            checks += 1
    return {"ok": True, "checks": checks, "sentinel": "n/a (dtype)",
            "max_err": 0.0}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            a = torch.empty(1024, dtype=torch.float16, device=p.torch_device)
            b = torch.empty(1024, dtype=torch.float32, device=p.torch_device)
            return lambda: fn(a, b)
        return _make

    from kernel.torch_level import result_type_torch
    return [PerfCase("ops.result_type.torch", group="ops", level="kernel",
                     make_fn=make(result_type_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
