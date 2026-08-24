#!/usr/bin/env python3
"""样例 B×框架层: vendor backend 拦截真实 vLLM 前向 + 黄金回归。

运行: python3 examples/b-framework/example.py [设备profile名]（约 2 分钟）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from common.device import load_profile
    from tests.framework_level import test_b

    profile = load_profile(
        sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print(f"[b-framework 样例] 设备: {profile.summary()}")
    print("  注入机制实配:")
    pkg, func = profile.delegate_import()
    print(f"    1. 插件: VLLM_FL_PLUGIN_MODULES=routes.b_vendor.backend.register_ops")
    print(f"    2. PER_OP 钉选: silu_and_mul=vendor:audit|reference（基线/插件同钉 reference）")
    print(f"    3. 委托: AUDIT_DELEGATE=reference; profile vendor_delegate={pkg}.{func}"
          f"（本机厂商 kernel 有缺陷故用 reference 委托保证恒等）\n")

    ok = test_b.run(profile)
    assert ok
    print("\n=> B×framework 样例 PASS")
    print("   同内容正式测试: python3 run.py --route b --level framework")


if __name__ == "__main__":
    main()
