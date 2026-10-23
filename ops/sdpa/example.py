#!/usr/bin/env python3
"""sdpa 一键编排: 三层测试一跑完 + 黄金精度入口提示。

运行: python3 example.py [ascend910|p800-kunlunxin|mlu590]
应用层为轻量消费方（mini-decoder，对齐 FlagOS-OP softmax-fullstack
先例——Python 层真实计算任务，不依赖 vLLM）
"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OP_DIR))


def main() -> None:
    from _profile import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[sdpa] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"sdpa_{name}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    for name, mod in [("1/3 算子库层(kernel 直测)", _load("kernel")),
                      ("2/3 框架层(A1 注册/拦截)", _load("op")),
                      ("3/3 应用层(mini-decoder 消费)", _load("framework"))]:
        print("\n" + "-" * 60)
        print(f"Stage {name}")
        print("-" * 60)
        r = mod.run(profile)
        print(f"  => PASS  {r}")

    print("\n[sdpa] 三层全绿。黄金精度与性能:")
    print("  python3 script/gen_golden.py            # CPU 生成（397 组）")
    print(f"  python3 script/check_accuracy.py --impl triton "
          f"--device {profile.torch_device}")
    print(f"  python3 script/bench_perf.py --device {profile.torch_device} "
          f"--json-out reports/perf_fp16_{profile.name}.json")


def _platform_tag(profile) -> str:
    """perf case id 平台标签。

    p800-kunlunxin 映射成历史标签 "p800"，保证入库基线
    ops.sdpa.p800.forward 不改名；其余平台直接用 profile 名
    （thead-ppu → ops.sdpa.thead-ppu.forward，与 embedding 先例一致）。
    """
    return {"p800-kunlunxin": "p800"}.get(profile.name, profile.name)


def perf_cases(profile):
    """Register the platform production path in the repository perf gate."""
    if profile.name not in ("p800-kunlunxin", "thead-ppu"):
        return []
    prefix = f"ops.sdpa.{_platform_tag(profile)}"

    from common.perf import PerfCase

    B, H, S, D = 1, 16, 1024, 128
    flops = 2 * B * H * S * S * D  # causal attention forward

    def make(fn):
        def _make(p):
            import torch

            torch.manual_seed(20260924)
            dtype = torch.float16
            q = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(
                p.torch_device
            )
            k = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(
                p.torch_device
            )
            v = (torch.randn(B, H, S, D) * 0.03).to(dtype).to(
                p.torch_device
            )

            def call():
                # 厂商委托 kernel 异步提交: 每次迭代读回一个元素强制完成
                # （known-issues #17.5 只测 launch 会得到物理不可能结果）。
                return fn(
                    q, k, v, None, 0.0, True, None, False
                ).reshape(-1)[0].item()

            return call

        return _make

    def tflops(ms):
        return {"TFLOPS": flops / (ms / 1000) / 1e12}

    import torch

    from ops.sdpa.kernel.triton_level import sdpa_triton

    def native(q, k, v, *args, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

    return [
        PerfCase(
            f"{prefix}.forward",
            group="ops", level="kernel",
            make_fn=make(sdpa_triton), derived=tflops,
        ),
        PerfCase(
            "ops.sdpa.native.forward",
            group="ops", level="kernel",
            make_fn=make(native), derived=tflops,
        ),
    ]


if __name__ == "__main__":
    main()
