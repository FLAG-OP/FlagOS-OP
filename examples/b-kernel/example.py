#!/usr/bin/env python3
"""样例 B×kernel 层: 厂商硬件语言 kernel 直测 + C++ JIT 编译闭环。

运行: python3 examples/b-kernel/example.py [设备profile名]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.kernel_level import test_b

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[b-kernel 样例] 设备: {profile.summary()}")
    print("  内容: JIT 编译闭环 + 厂商 kernel 直测 + 哨兵健全性检查\n")
    assert test_b.run(profile)
    print("\n=> B×kernel 样例 PASS")
    print("   正式测试: python3 run.py --route b --level kernel")


if __name__ == "__main__":
    main()
