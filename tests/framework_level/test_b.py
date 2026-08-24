#!/usr/bin/env python3
"""B 框架级: audit vendor 在真实 vLLM 前向中被调用，输出不变"""
from __future__ import annotations


def run(profile) -> bool:
    from tests.framework_level import _harness as H

    print("=" * 60)
    print(f"B framework [{profile.name}]: audit vendor interception")
    print("=" * 60)
    H.reset_counts("b")

    base = H.run_case("b", profile, "b_baseline.json", with_plugin=False)
    plug = H.run_case("b", profile, "b_plugin.json", with_plugin=True)
    counts = H.read_counts("b")

    print(f"  audit silu calls   : {counts.get('silu_and_mul', 0)}")
    assert counts.get("silu_and_mul", 0) > 0, "audit vendor 未被真实推理调用"

    # reference 委托（基线同样钉在 reference 路径）→ 数值恒等，
    # exact 断言成立。注: 本机 xtorch_ops.swiglu 独立调用不写 out，
    # 不能用于恒等断言（详见顶层 README 注意事项）。
    # exact 前缀按设备黄金逐 prompt 实测稳定性自适应
    cmp_res = H.compare_outputs(
        base, plug, exact=True, exact_prefix=8)
    print(f"  output compare     : {cmp_res['detail']}")
    assert cmp_res["ok"], "纯拦截下输出不应变化"

    golden_res = H.check_golden(base)
    print(f"  golden check       : {golden_res['detail']}")
    assert golden_res["ok"], "相对历史黄金输出发生漂移"
    print("  => B framework PASS")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"))
