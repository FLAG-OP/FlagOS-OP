# 性能回归用例模板（挂入 common/perf_registry.py 的 PROVIDERS）。
#
# 三方对比约定: 自研 / 原生 / FlagGems 各一条（FlagGems 无对应算子时注明）。
from __future__ import annotations


def perf_cases(profile):
    from common.perf import PerfCase
    from kernel import my_op_triton
    from reference import my_op_reference

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device) * 2
            g = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device)
            return lambda: fn(x, g)
        return _make

    def bw_3t(t):  # 读 x,g 写 out，按实际字节数调整
        return {"GBps": 8192 * 8192 * 2 * 3 / t / 1e6}

    return [
        PerfCase("example.my_op.triton", group="example", level="kernel",
                 make_fn=make(my_op_triton), derived=bw_3t),
        PerfCase("example.my_op.reference", group="example", level="kernel",
                 make_fn=make(my_op_reference)),
        # PerfCase("example.my_op.flaggems", ...)  # FlagGems 有对应算子时加
    ]
