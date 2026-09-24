# kernel 层直测（alias 类）: 共享存储 + requires_grad=False + 元数据。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    from kernel.torch_level import detach_torch
    from reference import detach_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    checks = 0
    for shape in [(64, 1024), (1, 14336), (8, 5000), (7, 4097)]:
        for dt in dts:
            for rg in (True, False):
                x = torch.randn(*shape, dtype=dt, device=dev)
                x.requires_grad_(rg)
                got = detach_torch(x)
                assert got.shape == x.shape and got.dtype == x.dtype
                assert got.stride() == tuple(x.stride())
                assert got.data_ptr() == x.data_ptr(), "未共享存储"
                assert got.requires_grad is False
                assert got is not x
                checks += 1
    x = torch.randn(256, 128, dtype=torch.float32, device=dev).t()
    got = detach_torch(x)
    assert got.data_ptr() == x.data_ptr() and got.stride() == x.stride()
    checks += 1
    return {"ok": True, "checks": checks, "sentinel": "n/a (alias)", "max_err": 0.0}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.float32,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    from kernel.torch_level import detach_torch
    return [PerfCase("ops.detach.torch", group="ops", level="kernel",
                     make_fn=make(detach_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
