#!/usr/bin/env python3
"""sigmoid 一键运行 kernel、op、应用三层验证。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _load_test(name: str):
    path = OP_DIR / "test" / f"{name}_level.py"
    spec = importlib.util.spec_from_file_location(f"sigmoid_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from common.device import load_profile
    profile = load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")
    print(f"[sigmoid] {profile.summary()}")
    for label, name in (("1/3 kernel 层", "kernel"),
                        ("2/3 op 层", "op"),
                        ("3/3 应用层", "framework")):
        print("\n" + "-" * 60)
        print(label)
        result = _load_test(name).run(profile)
        print(f"  => PASS {result}")
    print("\n[sigmoid] 三层验证通过。")


def perf_cases(profile):
    del profile
    return _load_test("kernel").perf_cases(None)


if __name__ == "__main__":
    main()
