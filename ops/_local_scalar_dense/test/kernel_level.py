# kernel 层直测（host 标量类）: 值/类型正确 + 多元素报错。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    from kernel.torch_level import _local_scalar_dense_torch

    dev = profile.torch_device
    checks = 0
    for dt in (torch.float32, torch.float16, torch.bfloat16,
               torch.int32, torch.int64, torch.bool):
        val = True if dt is torch.bool else (1.25 if dt.is_floating_point else 7)
        for shape in [(), (1,)]:
            x = torch.full(shape, val, dtype=dt, device=dev)
            got = _local_scalar_dense_torch(x)
            exp = x.item()
            assert got == exp, f"{dt} {shape}: {got} != {exp}"
            assert type(got) is type(exp)
            checks += 1
    try:
        _local_scalar_dense_torch(torch.randn(4, device=dev))
        raise AssertionError("多元素未报错")
    except RuntimeError:
        pass
    return {"ok": True, "checks": checks, "sentinel": "n/a (host 标量)",
            "max_err": 0.0}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(1, dtype=torch.float32, device=p.torch_device)
            return lambda: fn(x)
        return _make

    from kernel.torch_level import _local_scalar_dense_torch
    return [PerfCase("ops._local_scalar_dense.torch", group="ops", level="kernel",
                     make_fn=make(_local_scalar_dense_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
