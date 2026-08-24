#!/usr/bin/env python3
"""样例 A2×kernel 层: Triton gelu_and_mul 直测。

运行: python3 examples/a2-kernel/example.py [设备profile名]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.kernel_level import test_a2

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[a2-kernel 样例] 设备: {profile.summary()}")
    print("  内容: Triton gelu_and_mul 直测（精度+哨兵+性能）\n")
    assert test_a2.run(profile)
    print("\n=> A2×kernel 样例 PASS")
    print("   正式测试: python3 run.py --route a2 --level kernel")


if __name__ == "__main__":
    main()
