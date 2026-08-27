#!/usr/bin/env python3
"""样例 B-fullstack: 单一算子（silu_and_mul）贯穿三层——旗舰样例。

  算子库层: 写 C++ kernel → JIT 编译 → 直测（精度/哨兵/性能）
  框架层:   注册 vendor:my-cpp → PER_OP 钉选 → call_op 验证
  应用层:   注入真实 vLLM → 前向调用计数 + 输出比对 + 黄金回归

运行: python3 examples/b-fullstack/example.py [设备profile名]（约 3-4 分钟）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def stage_l0_kernel(profile) -> bool:
    """算子库层: JIT 编译 + 直测。"""
    import torch
    import torch.nn.functional as F

    dev = profile.torch_device
    print("-" * 60)
    print("Stage 1/3  kernel 层: C++ 源码 → JIT 编译 → 直测")
    print("-" * 60)

    sys.path.insert(0, str(HERE))
    from fullstack_plugin import _load_kernel

    t0 = time.perf_counter()
    ops = _load_kernel()
    print(f"  JIT 编译完成（{time.perf_counter() - t0:.1f}s，后续走 ninja 缓存）")

    def ref(x):
        d = x.shape[-1] // 2
        return (F.silu(x[..., :d].float()) * x[..., d:].float()).to(x.dtype)

    # 精度
    for shape in [(64, 1024), (128, 5120), (1, 14336)]:
        for dt in (torch.bfloat16, torch.float16):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            out = ops.silu_and_mul(x)
            err = (out.float() - ref(x).float()).abs().max().item()
            assert err < 1e-1, f"{shape} {dt} err={err}"
    print("  精度: 6/6 组合 PASS（fp32 内部计算）")

    # 哨兵
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    o1, o2 = ops.silu_and_mul(x), ops.silu_and_mul(x)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2) and not torch.equal(o1, ops.silu_and_mul(x + 1))
    print("  哨兵: 确定性 OK 输入敏感 OK")

    # 性能
    x = torch.randn(4096, 8192, dtype=torch.bfloat16, device=dev)
    for _ in range(20):
        ops.silu_and_mul(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        ops.silu_and_mul(x)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    print(f"  性能: {t:.3f}ms/call")
    return True


# ============ 性能回归用例（scripts/perf_run.py 消费） ============
def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        sys.path.insert(0, str(HERE))
        from fullstack_plugin import _load_kernel
        ops = _load_kernel()
        x = torch.randn(4096, 8192, dtype=torch.bfloat16,
                        device=p.torch_device)
        return lambda: ops.silu_and_mul(x)

    # 读 4096×8192 · 写 4096×4096（bf16）
    def bw(t):
        return {"GBps": (4096 * 8192 + 4096 * 4096) * 2 / t / 1e6}

    return [PerfCase("example.b-fullstack.silu_and_mul.cpp", group="example",
                     level="kernel", make_fn=make, derived=bw)]


def stage_l2_dispatch(profile) -> bool:
    """框架层: 注册 vendor:my-cpp + PER_OP 语义验证（进程内 with_allowed_vendors）。"""
    import torch

    dev = profile.torch_device
    print()
    print("-" * 60)
    print("Stage 2/3  op 层: vendor:my-cpp 注册 → dispatch 选择")
    print("-" * 60)

    os.environ["VLLM_FL_PLUGIN_MODULES"] = "fullstack_plugin"
    sys.path.insert(0, str(HERE))

    from vllm_fl.dispatch import get_default_manager, call_op, reset_default_manager
    from vllm_fl.dispatch.policy import with_preference, with_allowed_vendors

    reset_default_manager()
    m = get_default_manager()
    m.ensure_initialized()
    ids = [i.impl_id for i in
           m.registry.snapshot().impls_by_op.get("silu_and_mul", [])]
    assert "vendor.my-cpp" in ids, ids
    print(f"  注册: {ids}")

    x = torch.randn(64, 4096, dtype=torch.bfloat16, device=dev)
    with with_preference("vendor"), with_allowed_vendors("my-cpp"):
        call_op("silu_and_mul", None, x)
        used = m._called_ops["silu_and_mul"]
    assert used == "vendor.my-cpp", used
    print(f"  选择: {used}（with_allowed_vendors 精确钉住）")
    print("  （框架层将用 VLLM_FL_PER_OP=silu_and_mul=vendor:my-cpp|reference 达到同样效果）")
    return True


def stage_l4_framework(profile) -> bool:
    """应用层: 注入真实 vLLM（基线/插件双跑 + 计数 + 输出比对 + 黄金）。"""
    print()
    print("-" * 60)
    print("Stage 3/3  framework 验证: 真实 vLLM 前向注入")
    print("-" * 60)

    from tests.framework_level._harness import compare_outputs, check_golden

    count_base = "/tmp/my_cpp_counts"
    import glob as _glob
    for p in _glob.glob(f"{count_base}.*"):
        Path(p).unlink(missing_ok=True)

    def run(with_plugin: bool) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{HERE}{os.pathsep}{ROOT / 'injection'}{os.pathsep}{ROOT}"
        env["CUDA_VISIBLE_DEVICES"] = profile.visible_devices
        env["VLLM_LOGGING_LEVEL"] = "WARNING"
        env["PROMPTS_AS_TOKENS"] = "1"
        env["MY_CPP_COUNT_FILE"] = count_base
        env.pop("VLLM_FL_PLUGIN_MODULES", None)
        env.pop("VLLM_FL_PER_OP", None)
        # 基线钉 reference 路径（与黄金生成一致; 与 tests/framework_level/test_b 同模式）
        if not with_plugin:
            env["VLLM_FL_PER_OP"] = "silu_and_mul=reference"
        if with_plugin:
            env["VLLM_FL_PLUGIN_MODULES"] = "fullstack_plugin"
            env["VLLM_FL_PER_OP"] = "silu_and_mul=vendor:my-cpp|reference"

        out = HERE / ("fs_plugin.json" if with_plugin else "fs_baseline.json")
        cmd = [sys.executable,
               str(ROOT / "tests/framework_level/_vllm_runner.py"),
               str(out), "--device", profile.name, "--route", "b"]
        print(f"  running {'WITH' if with_plugin else 'WITHOUT'} plugin ...")
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        return json.loads(out.read_text())

    base = run(with_plugin=False)
    plug = run(with_plugin=True)

    total = {}
    for p in _glob.glob(f"{count_base}.*"):
        try:
            for k, v in json.load(open(p)).items():
                total[k] = total.get(k, 0) + v
        except (OSError, ValueError):
            pass
    print(f"  C++ kernel 真实前向调用: {total.get('silu_and_mul', 0)} 次")
    assert total.get("silu_and_mul", 0) > 0, "自研 C++ kernel 未被真实推理调用"

    # 自定义数值实现 → 前 2 token 一致率（同 A2 框架层标准）
    n = len(base["output_token_ids"])
    cmp = compare_outputs(base, plug, exact=False, k_tokens=2,
                          min_match_prompts=max(2, 2 * n // 3))
    print(f"  输出比对: {cmp['detail']}")
    assert cmp["ok"]

    gold = check_golden(base)
    print(f"  黄金回归: {gold['detail']}")
    assert gold["ok"]
    return True


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    print("=" * 60)
    print(f"B-fullstack 样例: silu_and_mul 贯穿 算子库层→框架层→应用层")
    print(f"设备: {profile.summary()}")
    print("=" * 60)

    assert stage_l0_kernel(profile)
    assert stage_l2_dispatch(profile)
    assert stage_l4_framework(profile)
    print()
    print("=> B-fullstack PASS: 同一 C++ kernel 在 kernel/dispatch/框架三层")
    print("   全部验证通过——这就是'算子在各层级自定义后在各层级验证'的完整故事线。")


if __name__ == "__main__":
    main()
