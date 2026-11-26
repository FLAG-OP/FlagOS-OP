#!/usr/bin/env python3
"""gelu 一键编排: 三层测试一次跑完 + 性能回归用例入口。

运行: python3 example.py [设备profile名]
（目录名 test/ 与标准库 test 包同名，故按路径加载规避导入遮蔽）
"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def main() -> None:
    from _profile import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")
    print(f"[gelu] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"gelu_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    all_ok = True
    for label, name in [("1/3 算子库层", "kernel"),
                        ("2/3 框架层", "op"),
                        ("3/3 应用层", "framework")]:
        print("\n" + "-" * 60)
        print(f"Stage {label}")
        print("-" * 60)
        r = _load(name).run(profile)
        ok = r.get("ok", False) if isinstance(r, dict) else bool(r)
        all_ok = all_ok and ok
        print(f"  => {label} {'PASS' if ok else 'FAIL'}  {r}")

    if not all_ok:
        raise SystemExit("[gelu] 存在失败层级")
    print("\n[gelu] 三层全绿。黄金与分项:")
    print("  python3 script/gen_golden.py --device cpu")
    print("  python3 script/check_accuracy.py --impl triton --device "
          + profile.torch_device)


def _load_by_path(rel: str, entry: str):
    """按显式路径加载本算子模块（perf_registry 多提供者防 sys.modules 串味）。"""
    import importlib.util
    p = OP_DIR / rel
    key = "gelu_" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(key, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def perf_cases(profile):
    """性能回归用例——挂入 common/perf_registry.py 的 PROVIDERS。"""
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(8192, 8192, dtype=torch.float16,
                            device=p.torch_device) * 2
            return lambda: fn(x, "tanh")
        return _make

    def bw(t):
        return {"GBps": 8192 * 8192 * 2 * 2 / t / 1e6}

    gelu_triton = _load_by_path("kernel/triton_level.py", "gelu_triton")
    gelu_torch = _load_by_path("kernel/torch_level.py", "gelu_torch")
    gelu_reference = _load_by_path("reference.py", "gelu_reference")
    gelu_ppu = _load_by_path("kernel/thead_ppu.py", "gelu_ppu")

    return [
        PerfCase("ops.gelu.triton", group="ops", level="kernel",
                 make_fn=make(gelu_triton), derived=bw),
        PerfCase("ops.gelu.native", group="ops", level="kernel",
                 make_fn=make(gelu_ppu), derived=bw),
        PerfCase("ops.gelu.torch", group="ops", level="kernel",
                 make_fn=make(gelu_torch), derived=bw),
        PerfCase("ops.gelu.reference", group="ops", level="kernel",
                 make_fn=make(gelu_reference), derived=bw),
    ]


if __name__ == "__main__":
    main()
