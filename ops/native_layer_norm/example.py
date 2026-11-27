#!/usr/bin/env python3
"""native_layer_norm 一键编排: 三层测试一次跑完 + 性能回归用例入口。

运行: python3 example.py [设备profile名]
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
    print(f"[native_layer_norm] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"nln_{name}", p)
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
        raise SystemExit("[native_layer_norm] 存在失败层级")
    print("\n[native_layer_norm] 三层全绿。黄金与分项:")
    print("  python3 script/gen_golden.py --device cpu")
    print("  python3 script/check_accuracy.py --impl native --device "
          + profile.torch_device)


def _load_by_path(rel: str, entry: str):
    import importlib.util
    p = OP_DIR / rel
    key = "nln_" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(key, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(4096, 4096, dtype=torch.float16,
                            device=p.torch_device)
            w = torch.randn(4096, dtype=torch.float16,
                            device=p.torch_device)
            b = torch.randn(4096, dtype=torch.float16,
                            device=p.torch_device)
            return lambda: fn(x, (4096,), w, b, 1e-5)
        return _make

    def bw(t):
        return {"GBps": 4096 * 4096 * 2 * 2 / t / 1e6}

    nln_ppu = _load_by_path("kernel/thead_ppu.py", "native_layer_norm_ppu")
    nln_torch = _load_by_path("kernel/torch_level.py",
                              "native_layer_norm_torch")
    nln_ref = _load_by_path("reference.py", "native_layer_norm_reference")

    return [
        PerfCase("ops.native_layer_norm.native", group="ops", level="kernel",
                 make_fn=make(nln_ppu), derived=bw),
        PerfCase("ops.native_layer_norm.torch", group="ops", level="kernel",
                 make_fn=make(nln_torch), derived=bw),
        PerfCase("ops.native_layer_norm.reference", group="ops",
                 level="kernel", make_fn=make(nln_ref), derived=bw),
    ]


if __name__ == "__main__":
    main()
