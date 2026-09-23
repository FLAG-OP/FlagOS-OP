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
    print("  python3 script/gen_golden.py            # CPU 生成（265 组）")
    print(f"  python3 script/check_accuracy.py --impl triton "
          f"--device {profile.torch_device}")
    print(f"  python3 script/bench_perf.py --device {profile.torch_device} "
          f"--json-out reports/perf_fp16_{profile.name}.json")


if __name__ == "__main__":
    main()
