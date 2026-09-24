# kernel 层直测（分配算子）: 按指定 stride 的元数据 + 重叠/零元素。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    from kernel.torch_level import empty_strided_torch
    from reference import empty_strided_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    checks = 0

    cases = []
    for shape in [(64, 1024), (1, 14336), (8, 5000), (7, 4097)]:
        strides = [shape[1], 1]
        for dt in dts:
            cases.append((shape, strides, dt))
    cases += [((128, 256), [1, 128], torch.float32),        # 转置型
              ((4, 8, 16), [128, 16, 1], torch.float32),    # 3D
              ((4, 4), [1, 0], torch.float32),              # 重叠 stride
              ((0, 5), [5, 1], torch.float32)]              # 零元素

    for shape, strides, dt in cases:
        got = empty_strided_torch(shape, strides, dtype=dt, device=dev)
        exp = empty_strided_reference(shape, strides, dtype=dt)
        assert tuple(got.shape) == tuple(shape), f"{shape} shape"
        assert got.dtype == dt
        assert got.stride() == tuple(strides), \
            f"{shape} {strides} stride {got.stride()}"
        assert got.stride() == exp.stride()
        checks += 1

    return {"ok": True, "checks": checks, "sentinel": "n/a (分配算子)",
            "max_err": 0.0}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            return lambda: fn((8192, 8192), [8192, 1],
                              dtype=torch.float32, device=p.torch_device)
        return _make

    from kernel.torch_level import empty_strided_torch
    return [PerfCase("ops.empty_strided.torch", group="ops", level="kernel",
                     make_fn=make(empty_strided_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
