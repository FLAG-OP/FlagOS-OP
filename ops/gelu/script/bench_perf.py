#!/usr/bin/env python3
"""gelu 性能测试: 短采样 + 同步 + 独立口径（issue #55 交付要求 4）。

口径（docs/acceptance.md 性能测量口径）:
  - warmup 20 + iters ≤100（#10 分配器陷阱: 不用长循环）
  - 计时区段前后 synchronize（防假异步计时）
  - 四方对照: 自研 triton / 原生 native / torch 级 / fp32 reference
    （原生/reference 参与对照, 输出设备执行时间与 GBps）

用法:
  python3 script/bench_perf.py --device cuda:6 \
      --json-out reports/perf_fp16_thead-ppu.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
SHAPES = [(4096, 4096), (8192, 8192), (1, 5000), (128, 5120)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)

    triton_mod = _load("gelu_triton", "kernel/triton_level.py")
    ppu_mod = _load("gelu_ppu", "kernel/thead_ppu.py")
    torch_mod = _load("gelu_torch", "kernel/torch_level.py")
    ref_mod = _load("gelu_reference", "reference.py")

    impls = {
        "triton": lambda x: triton_mod.gelu_triton(x, "tanh"),
        "native": lambda x: ppu_mod.gelu_ppu(x, "tanh"),
        "torch": lambda x: torch_mod.gelu_torch(x, "tanh"),
        "reference": lambda x: ref_mod.gelu_reference(x, "tanh"),
    }

    rows = []
    for shape in SHAPES:
        for dt in DTYPES:
            x = (torch.randn(*shape, dtype=torch.float32, device=dev) * 2
                 ).to(dt)
            nbytes = x.numel() * x.element_size() * 2  # 读 + 写
            row = {"shape": list(shape), "dtype": str(dt).replace("torch.", "")}
            for name, fn in impls.items():
                try:
                    t_ms = _bench(lambda: fn(x))
                    row[f"{name}_ms"] = round(t_ms, 4)
                    row[f"{name}_GBps"] = round(nbytes / t_ms / 1e6, 2)
                except Exception as e:  # 对照路径缺栈时记缺证不静默
                    row[f"{name}_ms"] = None
                    row[f"{name}_error"] = repr(e)[:120]
            if row.get("native_ms") and row.get("triton_ms"):
                row["triton_vs_native"] = round(
                    row["native_ms"] / row["triton_ms"], 3)
            rows.append(row)
            print(row)

    result = {"op": "gelu", "device": str(dev),
              "device_name": torch.cuda.get_device_name(dev),
              "method": "warmup20+iters100+synchronize, tanh mode",
              "rows": rows}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
