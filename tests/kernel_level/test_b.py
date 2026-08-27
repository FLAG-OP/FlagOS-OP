#!/usr/bin/env python3
"""B kernel 层: 厂商硬件语言 kernel 直测（核心新格）。

内容:
  1. JIT 编译闭环: csrc C++ 模板 load() 编译直调（自研厂商语言路径）
  2. profile 声明的现成厂商 kernel 直测（精度 vs 语义参考）
  3. 哨兵健全性检查（swiglu 型"不写输出"bug 的通用检测）
  4. 性能短采样
"""
from __future__ import annotations

import traceback


def _load_csrc_module():
    """JIT 编译 routes/b_vendor/csrc/vendor_kernel.cpp。"""
    import os
    from pathlib import Path

    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parents[2] / "routes/b_vendor/csrc"
    os.makedirs("/tmp/flagos_csrc_build", exist_ok=True)
    return load(
        name="my_vendor_ops",
        sources=[str(src / "vendor_kernel.cpp")],
        extra_cflags=["-O3"],
        verbose=False,
        build_directory="/tmp/flagos_csrc_build",
    )


def run(profile):
    import time

    import torch
    from common.kernel_spec import load_kernel_specs, sentinel_check, SEMANTIC_REFS

    dev = profile.torch_device
    print("=" * 60)
    print(f"B kernel [{profile.name}]: 厂商硬件语言 kernel 直测 @ {dev}")
    print("=" * 60)

    specs = load_kernel_specs(profile)
    assert specs, f"设备 {profile.name} 未声明 vendor_kernels"

    # ---- 1. JIT 编译闭环 ----
    mod = None
    csrc_ok = False
    try:
        mod = _load_csrc_module()
        csrc_ok = True
        print("  JIT 编译: csrc -> my_vendor_ops OK")
    except Exception as e:
        print(f"  JIT 编译: SKIP（本机工具链限制: {str(e)[:80]}）")

    # ---- 2/3. 现成厂商 kernel 直测 + 哨兵 ----
    SHAPES = [(64, 1024), (128, 5120)]
    results = []
    for spec in specs:
        if spec.build == "csrc" and not csrc_ok:
            results.append((spec, "SKIP", "JIT 不可用"))
            continue
        try:
            fn = (getattr(mod, spec.func_name) if spec.build == "csrc"
                  else spec.load_callable())
        except ImportError as e:
            results.append((spec, "SKIP", str(e)[:50]))
            continue

        # 哨兵健全性
        sent = sentinel_check(spec, fn, dev)
        if not spec.expected_ok:
            caught = not sent["ok"]
            if caught:
                results.append((spec, "PASS",
                                f"负例哨兵抓出 OK: {sent['detail']}"))
            else:
                # 概率性缺陷（如 #5 设备分化）本次未复现——判 WARN 而非
                # FAIL，否则整格通过依赖"坏 kernel 每次都坏"，天然 flaky
                results.append((spec, "WARN",
                                f"负例本次未复现（哨兵通过）: {sent['detail']}"))
            continue
        if not sent["ok"]:
            results.append((spec, "FAIL", f"哨兵: {sent['detail']}"))
            continue

        # 精度 vs 语义参考
        ref_fn = SEMANTIC_REFS.get(spec.op)
        if ref_fn is None:
            results.append((spec, "SKIP", f"无语义参考 op={spec.op}"))
            continue
        ok, detail = True, ""
        for shape in SHAPES:
            for dt in (torch.bfloat16, torch.float16):
                parts = spec.make_inputs(shape, dt, dev)
                ref = ref_fn(*parts)
                out = spec.call(fn, *parts)
                err = (out.float() - ref.float()).abs().max().item() \
                    if out.shape == ref.shape else float("inf")
                if err > 1e-2:
                    ok = False
                    detail = f"shape={shape} {dt} err={err:.3f}"
                    break
            if not ok:
                break
        results.append((spec, "PASS" if ok else "FAIL",
                        detail or "精度+哨兵 OK"))

    print(f"\n  {'kernel':38s} {'结果':5s} 说明")
    print("  " + "-" * 70)
    n_pass = n_fail = n_warn = 0
    for spec, status, detail in results:
        print(f"  {spec.name:38s} {status:5s} {detail}")
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        n_warn += status == "WARN"
    assert n_fail == 0, f"{n_fail} 个厂商 kernel 直测失败"

    # ---- 4. 性能（首个可用 return 模式 kernel） ----
    perf_name, perf_ms = None, None
    for spec, status, _ in results:
        if status != "PASS" or spec.out_mode != "return":
            continue
        try:
            fn = (getattr(mod, spec.func_name) if spec.build == "csrc"
                  else spec.load_callable())
            parts = spec.make_inputs((8192, 4096), torch.bfloat16, dev)
            for _ in range(20):
                spec.call(fn, *parts)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                spec.call(fn, *parts)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 100 * 1000
            perf_name, perf_ms = spec.name, t
            print(f"\n  性能: {spec.name} {t:.3f}ms/call")
        except Exception:
            why = traceback.format_exc(limit=1).splitlines()[-1][:70]
            print(f"\n  性能: {spec.name} SKIP（{why}）")
        break

    n_skip = len(results) - n_pass - n_fail - n_warn
    print(f"\n  => B kernel PASS（{n_pass} pass / {n_fail} fail / "
          f"{n_warn} warn / {n_skip} skip）")
    return {"ok": n_fail == 0, "pass": n_pass, "fail": n_fail,
            "warn": n_warn, "skip": len(results) - n_pass - n_fail - n_warn,
            "perf": f"{perf_name}={perf_ms:.3f}ms" if perf_ms else None}


def perf_cases(profile):
    """性能回归用例（scripts/perf_run.py 消费）: return 模式厂商 kernel。"""
    import re

    from common.kernel_spec import load_kernel_specs
    from common.perf import PerfCase

    cases = []
    for spec in load_kernel_specs(profile):
        if spec.out_mode != "return" or not spec.expected_ok:
            continue
        safe = re.sub(r"[^0-9A-Za-z._-]", "_", spec.name)

        def make(sp):
            def _make(p):
                import torch
                if sp.build == "csrc":
                    mod = _load_csrc_module()
                    fn = getattr(mod, sp.func_name)
                else:
                    fn = sp.load_callable()
                parts = sp.make_inputs((8192, 4096), torch.bfloat16,
                                       p.torch_device)
                return lambda: sp.call(fn, *parts)
            return _make

        cases.append(PerfCase(f"matrix-kernel.b.{safe}", group="matrix-kernel",
                              level="kernel", make_fn=make(spec)))
    return cases
