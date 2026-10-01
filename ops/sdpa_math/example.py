#!/usr/bin/env python3
"""sdpa_math 一键编排: 三层测试一跑完 + 黄金精度入口提示。

运行: python3 example.py [ascend910|cpu|p800-kunlunxin]
应用层为轻量消费方（mini-decoder + 概率图消费者，对齐 FlagOS-OP
softmax-fullstack 先例——Python 层真实计算任务，不依赖 vLLM）。
"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def main() -> None:
    from _profile import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[sdpa_math] {profile.summary()}")

    import importlib.util

    def _load(name):
        p = OP_DIR / "test" / f"{name}_level.py"
        spec = importlib.util.spec_from_file_location(f"sdpa_math_{name}", p)
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

    print("\n[sdpa_math] 三层全绿。黄金精度与性能:")
    print("  python3 script/gen_golden.py            # CPU 生成（175 组）")
    print(f"  python3 script/check_accuracy.py --impl "
          f"{profile.default_impl} "
          f"--device {profile.torch_device}")
    print(f"  python3 script/check_accuracy.py --impl native "
          f"--device {profile.torch_device}   # 原生对照（bool 用例跳过）")
    print(f"  python3 script/bench_perf.py --device {profile.torch_device} "
          f"--register --json-out reports/perf_{profile.name}.json")
    print("  python3 probes/native_semantics.py      # native 语义证据表")


def _load_by_path(rel: str, entry: str):
    """按路径加载（唯一模块名）——perf 门禁在同进程里加载多个算子的
    example.py，直接 `from kernel... import` 会撞到别的算子已注册的同名
    模块（softmax-fullstack 的 kernel.triton_level 实测踩过）。"""
    import importlib.util
    key = "sdpa_math_" + rel.replace("/", "_")[:-3]
    spec = importlib.util.spec_from_file_location(key, OP_DIR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def perf_cases(profile):
    """性能回归用例——挂入 common/perf_registry.py 的 PROVIDERS。"""
    from common.perf import PerfCase

    B, H, S, D = 1, 16, 1024, 128
    flops = 2 * B * H * S * S * D   # causal attention ≈ 全量 QK/PV 的一半

    def make(fn):
        def _make(p):
            import torch
            g = torch.Generator(device="cpu").manual_seed(20260930)
            q = (torch.randn(B, H, S, D, generator=g) * 0.05).to(
                torch.float16).to(p.torch_device)
            k = (torch.randn(B, H, S, D, generator=g) * 0.05).to(
                torch.float16).to(p.torch_device)
            v = (torch.randn(B, H, S, D, generator=g) * 0.05).to(
                torch.float16).to(p.torch_device)

            def call():
                # 读一个输出元素: 防异步栈只测到提交（#10 同类教训）
                return fn(q, k, v, None, 0.0, True, None)[0].reshape(-1)[
                    0].item()
            return call
        return _make

    def tflops(ms):
        return {"TFLOPS": flops / (ms / 1000) / 1e12}

    torch_fn = _load_by_path("kernel/torch_level.py", "sdpa_math_torch")
    ref = _load_by_path("reference.py", "sdpa_math_reference")
    if profile.vendor == "kunlunxin":
        import torch
        p800_fn = _load_by_path(
            "kernel/p800_fast_level.py", "sdpa_math_p800_fast")

        def native(q, k, v, *args, **kwargs):
            return torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, *args, **kwargs)
        return [
            PerfCase("ops.sdpa_math.p800", group="ops", level="kernel",
                     make_fn=make(p800_fn), derived=tflops, iters=50),
            PerfCase("ops.sdpa_math.native", group="ops", level="kernel",
                     make_fn=make(native), derived=tflops, iters=50),
        ]

    triton = _load_by_path("kernel/triton_level.py", "sdpa_math_triton")
    return [
        PerfCase("ops.sdpa_math.triton", group="ops", level="kernel",
                 make_fn=make(triton), derived=tflops, iters=50),
        PerfCase("ops.sdpa_math.torch", group="ops", level="kernel",
                 make_fn=make(torch_fn), derived=tflops, iters=50),
        PerfCase("ops.sdpa_math.reference", group="ops", level="kernel",
                 make_fn=make(ref), derived=tflops, iters=50),
    ]


if __name__ == "__main__":
    main()
