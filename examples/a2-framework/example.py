#!/usr/bin/env python3
"""样例 A2×框架层: Triton 实现注入真实 vLLM 前向。

机制: Triton silu_and_mul 以 vendor:triton-template 身份注册
（PER_OP 支持 vendor:<name> 精确钉住），PLUGIN_MODULES 注入 + 钉选。

运行: python3 examples/a2-framework/example.py [设备profile名]（约 2 分钟）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.framework_level import test_a2

    profile = load_profile(
        sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[a2-framework 样例] 设备: {profile.summary()}")
    print("  注入: vendor:triton-template + PER_OP 精确钉住")
    print("  断言: 计数>0 / 前2token一致率(自定义数值允许混沌分叉)\n")

    ok = test_a2.run(profile)
    assert ok
    print("\n=> A2×framework 样例 PASS")
    print("   同内容正式测试: python3 run.py --route a2 --level framework")


if __name__ == "__main__":
    main()
