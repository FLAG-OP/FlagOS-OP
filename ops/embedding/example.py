#!/usr/bin/env python3
"""Run all three embedding validation levels."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _load_level(name: str):
    path = OP_DIR / "test" / f"{name}_level.py"
    spec = importlib.util.spec_from_file_location(
        f"embedding_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from _profile import load_profile

    profile = load_profile(
        sys.argv[1] if len(sys.argv) > 1 else None
    )
    print(f"[embedding] {profile.summary()}")
    for title, name in [
        ("1/3 kernel", "kernel"),
        ("2/3 A1 op", "op"),
        ("3/3 application", "framework"),
    ]:
        print("\n" + "-" * 64)
        print(title)
        print("-" * 64)
        result = _load_level(name).run(profile)
        print(f"=> PASS {result}")

    print("\n[embedding] three levels passed. Goldens and performance:")
    print("  python3 script/gen_golden.py")
    print(
        "  python3 script/check_accuracy.py --impl p800 "
        f"--device {profile.torch_device}"
    )
    print(
        "  python3 script/bench_perf.py "
        f"--device {profile.torch_device} --json-out "
        "reports/perf_fp16_p800-kunlunxin.json"
    )


if __name__ == "__main__":
    main()
