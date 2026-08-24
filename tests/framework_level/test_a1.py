#!/usr/bin/env python3
"""A1 框架级: 数值恒等 aten::silu 在真实 vLLM 前向中被调用，输出不变"""
from __future__ import annotations


def run(profile) -> bool:
    from tests.framework_level import _harness as H

    print("=" * 60)
    print(f"A1 framework [{profile.name}]: aten::silu identity interception")
    print("=" * 60)
    H.reset_counts("a1")

    base = H.run_case("a1", profile, "a1_baseline.json", with_plugin=False)
    plug = H.run_case("a1", profile, "a1_plugin.json", with_plugin=True)
    counts = H.read_counts("a1")

    print(f"  aten::silu calls   : {counts.get('silu', 0)}")
    assert counts.get("silu", 0) > 0, "aten::silu 未被真实推理调用"

    # exact 前缀按设备黄金逐 prompt 实测稳定性自适应
    cmp_res = H.compare_outputs(
        base, plug, exact=True, exact_prefix=8)
    print(f"  output compare     : {cmp_res['detail']} [{cmp_res['mode']}]")
    assert cmp_res["ok"], "数值恒等覆盖下输出不应变化"

    golden_res = H.check_golden(base)
    print(f"  golden check       : {golden_res['detail']}")
    assert golden_res["ok"], "相对历史黄金输出发生漂移"
    print("  => A1 framework PASS")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"))
