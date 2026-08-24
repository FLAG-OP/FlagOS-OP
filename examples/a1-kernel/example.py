#!/usr/bin/env python3
"""样例 A1×kernel 层: Triton kernel 直测（不经任何注册/分发）。

运行: python3 examples/a1-kernel/example.py [设备profile名]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.kernel_level import test_a1

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[a1-kernel 样例] 设备: {profile.summary()}")
    print("  内容: Triton gelu 直测（精度 vs CPU 参考 + 哨兵 + 性能）\n")
    assert test_a1.run(profile)
    print("\n=> A1×kernel 样例 PASS")
    print("   正式测试: python3 run.py --route a1 --level kernel")


if __name__ == "__main__":
    main()
