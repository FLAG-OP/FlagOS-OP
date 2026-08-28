#!/usr/bin/env python3
"""样例 softmax-fullstack: 行归约算子贯穿三层（样板同构组织）。

目录结构与 templates/operator/ 一致——kernel/ 三级实现、test/ 三层
测试、goldendata/、script/、REPORT.md + reports/；硬件级置空并说明
（kernel/hardware_level/README.md）。

运行: python3 examples/softmax-fullstack/example.py [设备profile名]
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
    print(f"[softmax-fullstack] {profile.summary()}")

    # 注: 目录名 test/ 与标准库 test 包同名，按路径加载规避导入遮蔽
    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"smx_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    k, o, f = _load("kernel"), _load("op"), _load("framework")

    for name, mod in [("1/3 算子库层", k), ("2/3 框架层", o),
                      ("3/3 应用层", f)]:
        print("\n" + "-" * 60)
        print(f"Stage {name}")
        print("-" * 60)
        r = mod.run(profile)
        print(f"  => {name} PASS  {r}")

    print("\n[softmax-fullstack] 三层全绿。黄金与分项脚本:")
    print("  python3 script/gen_golden.py --device cpu")
    print("  python3 script/check_accuracy.py --impl triton --device "
          + profile.name)


def perf_cases(profile):
    from common.perf import PerfCase

    from flag_gems import ops as FG
    from kernel.torch_level import softmax_torch
    from kernel.triton_level import softmax_triton

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(1024, 1024, dtype=torch.bfloat16,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    def bw_2t(t):
        return {"GBps": 1024 * 1024 * 2 * 2 / t / 1e6}

    return [
        PerfCase("example.softmax-fullstack.softmax.triton", group="example",
                 level="kernel", make_fn=make(softmax_triton), derived=bw_2t),
        PerfCase("example.softmax-fullstack.softmax.torch", group="example",
                 level="kernel", make_fn=make(softmax_torch)),
        PerfCase("example.softmax-fullstack.softmax.flaggems", group="example",
                 level="kernel", make_fn=make(lambda x: FG.softmax(x, -1))),
    ]


if __name__ == "__main__":
    main()
