#!/usr/bin/env python3
"""spmm 一键编排: 三层测试一次跑完 + 性能回归用例入口。

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
    print(f"[spmm] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"spmm_{name}", p)
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
        raise SystemExit("[spmm] 存在失败层级")
    print("\n[spmm] 三层全绿。原生缺口取证: probes/thead_dense_p0_probe.json")
    print("  python3 script/gen_golden.py --device cpu")


def _load_by_path(rel: str, entry: str):
    import importlib.util
    p = OP_DIR / rel
    key = "spmm_" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(key, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            m, k, n, dens = 8192, 64, 64, 0.01
            dense = torch.rand(m, k, device=p.torch_device) * (torch.rand(
                m, k, device=p.torch_device) < dens)
            a = dense.to_sparse_csr()
            b = torch.randn(k, n, device=p.torch_device)
            return lambda: fn(a, b)
        return _make

    def gf(t):
        return {"GFLOPS": 2 * 8192 * 64 * 0.01 * 64 / t / 1e6}

    spmm_ppu = _load_by_path("kernel/thead_ppu.py", "spmm_ppu")
    spmm_reference = _load_by_path("reference.py", "spmm_reference")

    return [
        PerfCase("ops.spmm.index_add", group="ops", level="kernel",
                 make_fn=make(spmm_ppu), derived=gf),
        PerfCase("ops.spmm.reference", group="ops", level="kernel",
                 make_fn=make(spmm_reference), derived=gf),
    ]


if __name__ == "__main__":
    main()
