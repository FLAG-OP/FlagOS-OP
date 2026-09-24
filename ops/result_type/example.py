#!/usr/bin/env python3
"""result_type 一键编排: 三层测试一次跑完 + 性能回归用例入口。"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")
    print(f"[result_type] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"result_type_{name}", p)
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
        raise SystemExit("[result_type] 存在失败层级")
    print("\n[result_type] 三层全绿。")


def _load_by_path(rel, entry):
    import importlib.util
    p = OP_DIR / rel
    key = "result_type" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(key, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            a = torch.empty(1024, dtype=torch.float16, device=p.torch_device)
            b = torch.empty(1024, dtype=torch.float32, device=p.torch_device)
            return lambda: fn(a, b)
        return _make

    torch_fn = _load_by_path("kernel/torch_level.py", "result_type_torch")
    ref = _load_by_path("reference.py", "result_type_reference")
    return [PerfCase("ops.result_type.torch", group="ops", level="kernel",
                     make_fn=make(torch_fn)),
            PerfCase("ops.result_type.reference", group="ops", level="kernel",
                     make_fn=make(ref))]


if __name__ == "__main__":
    main()
