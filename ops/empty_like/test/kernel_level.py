# kernel 层直测（元数据算子）: shape/dtype/stride/存储独立。
# 无数值语义（未初始化），故无精度/哨兵判定。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    from kernel.torch_level import empty_like_torch
    from reference import empty_like_reference

    dev = profile.torch_device
    dts = (torch.bfloat16, torch.float16, torch.float32)
    n_checks = 0

    for shape in [(64, 1024), (1, 14336), (8, 5000), (7, 4097)]:
        for dt in dts:
            base = torch.randn(*shape, dtype=dt, device=dev)
            for view in ("contig", "transpose"):
                self_t = base if view == "contig" else base.transpose(0, 1)
                for mf in (torch.preserve_format, torch.contiguous_format):
                    got = empty_like_torch(self_t, memory_format=mf)
                    exp = empty_like_reference(self_t, memory_format=mf)
                    assert got.shape == exp.shape, f"{shape} {dt} {view} shape"
                    assert got.dtype == exp.dtype, f"{shape} {dt} {view} dtype"
                    assert got.stride() == exp.stride(), \
                        f"{shape} {dt} {view} {mf} stride {got.stride()} != {exp.stride()}"
                    assert got.data_ptr() != self_t.data_ptr(), "共享存储"
                    n_checks += 1

    # channels_last 分支
    cl = torch.randn(2, 3, 16, 16, dtype=torch.float32, device=dev).to(
        memory_format=torch.channels_last)
    got = empty_like_torch(cl, memory_format=torch.preserve_format)
    assert got.stride() == cl.stride(), "channels_last preserve stride"
    n_checks += 1

    # 零元素
    z = torch.empty(0, 5, dtype=torch.float32, device=dev)
    got = empty_like_torch(z)
    assert got.shape == (0, 5) and got.numel() == 0, "零元素"
    n_checks += 1

    # 三方一致
    x = torch.randn(512, 128, dtype=torch.float16, device=dev).t()
    assert empty_like_torch(x).stride() == empty_like_reference(x).stride()

    return {"ok": True, "checks": n_checks,
            "sentinel": "n/a (分配算子)", "max_err": 0.0,
            "latency_ms": None}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.float32,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    from kernel.torch_level import empty_like_torch
    return [PerfCase("ops.empty_like.torch", group="ops", level="kernel",
                     make_fn=make(empty_like_torch))]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
