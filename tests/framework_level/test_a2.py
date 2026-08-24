#!/usr/bin/env python3
"""A2 框架级: Triton silu_and_mul(vendor:triton-template) 注入真实 vLLM"""
from __future__ import annotations


def run(profile) -> bool:
    from tests.framework_level import _harness as H

    print("=" * 60)
    print(f"A2 framework [{profile.name}]: Triton silu_and_mul via dispatch")
    print("=" * 60)
    H.reset_counts("a2")

    base = H.run_case("a2", profile, "a2_baseline.json", with_plugin=False)
    plug = H.run_case("a2", profile, "a2_plugin.json", with_plugin=True)
    counts = H.read_counts("a2")

    print(f"  triton silu calls  : {counts.get('silu_and_mul', 0)}")
    assert counts.get("silu_and_mul", 0) > 0, "Triton 实现未被真实推理调用"

    # 自定义数值实现 → 前 2 token 一致率断言（允许混沌分叉）
    n_prompts = len(base["output_token_ids"])
    min_match = max(2, 2 * n_prompts // 3)
    cmp_res = H.compare_outputs(base, plug, exact=False, k_tokens=2,
                                min_match_prompts=min_match)
    print(f"  output compare     : {cmp_res['detail']}")
    assert cmp_res["ok"], "前 2 token 一致率过低（应 ≥2/3）"

    print("  => A2 framework PASS")
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from common.device import load_profile
    run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"))
