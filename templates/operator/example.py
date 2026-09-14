#!/usr/bin/env python3
"""my_op 一键编排: 三层测试一次跑完 + 性能回归用例入口。

运行: python3 example.py [设备profile名]
（与 softmax-fullstack 的编排器同构；目录名 test/ 与标准库 test 包
同名，故按路径加载规避导入遮蔽）
"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")
    print(f"[my_op] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"myop_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    for name, mod in [("1/3 算子库层", _load("kernel")),
                      ("2/3 框架层", _load("op")),
                      ("3/3 应用层", _load("framework"))]:
        print("\n" + "-" * 60)
        print(f"Stage {name}")
        print("-" * 60)
        r = mod.run(profile)
        print(f"  => {name} PASS  {r}")

    print("\n[my_op] 三层全绿。黄金与分项:")
    print("  python3 script/gen_golden.py --device cpu")
    print("  python3 script/check_accuracy.py --impl triton --device "
          + profile.name)


def perf_cases(profile):
    """性能回归用例——挂入 common/perf_registry.py 的 PROVIDERS。"""
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device) * 2
            g = torch.randn(8192, 8192, dtype=torch.bfloat16,
                            device=p.torch_device)
            return lambda: fn(x, g)
        return _make

    def bw_3t(t):  # 读 x,g 写 out（按实际字节调整）
        return {"GBps": 8192 * 8192 * 2 * 3 / t / 1e6}

    from kernel.torch_level import my_op_torch
    from kernel.triton_level import my_op_triton
    from reference import my_op_reference

    return [
        PerfCase("example.my_op.triton", group="example", level="kernel",
                 make_fn=make(my_op_triton), derived=bw_3t),
        PerfCase("example.my_op.reference", group="example", level="kernel",
                 make_fn=make(my_op_reference)),
        PerfCase("example.my_op.torch", group="example", level="kernel",
                 make_fn=make(my_op_torch)),
        # PerfCase("example.my_op.flaggems", ...)  # 有同语义生产实现时加
    ]


if __name__ == "__main__":
    main()
