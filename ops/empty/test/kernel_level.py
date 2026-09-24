# kernel 层直测（分配算子）: size/dtype/stride/memory_format 元数据。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    from kernel.torch_level import empty_torch
    from reference import empty_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    checks = 0

    for shape in [(64, 1024), (1, 14336), (8, 5000), (7, 4097)]:
        for dt in dts:
            got = empty_torch(shape, dtype=dt, device=dev)
            exp = empty_reference(shape, dtype=dt)
            assert tuple(got.shape) == shape and got.dtype == dt
            assert got.stride() == exp.stride(), \
                f"{shape} {dt} stride {got.stride()} != {exp.stride()}"
            checks += 1

    # channels_last
    cl = empty_torch((2, 3, 16, 16), dtype=torch.float32, device=dev,
                     memory_format=torch.channels_last)
    cl_ref = empty_reference((2, 3, 16, 16), dtype=torch.float32,
                             memory_format=torch.channels_last)
    assert cl.stride() == cl_ref.stride(), "channels_last stride"
    checks += 1

    # 零元素 + 默认 dtype
    z = empty_torch((0, 5), dtype=torch.float32, device=dev)
    assert z.shape == (0, 5) and z.numel() == 0
    d = empty_torch((4, 4), device=dev)
    assert d.dtype == torch.get_default_dtype()
    checks += 2

    return {"ok": True, "checks": checks, "sentinel": "n/a (分配算子)",
            "max_err": 0.0}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            return lambda: fn((8192, 8192), dtype=torch.float32,
                              device=p.torch_device)
        return _make

    from kernel.torch_level import empty_torch
    return [PerfCase("ops.empty.torch", group="ops", level="kernel",
                     make_fn=make(empty_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
