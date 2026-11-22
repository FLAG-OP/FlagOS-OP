#!/usr/bin/env python3
"""独立子进程运行 sigmoid 的尺寸 × dtype × 实现性能矩阵。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>
# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>

CASES = [
    ("tail_hidden", [1, 14336]),
    ("small_activation", [64, 1024]),
    ("transformer", [2, 64, 4096]),
    ("throughput", [2048, 2048]),
]
DTYPES = ["float16", "bfloat16", "float32"]
IMPLS = ["triton", "flaggems", "native"]


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_one(device: str, case_name: str, shape: list[int],
             dtype: str, impl: str, warmup: int, iters: int) -> dict:
    command = [
        sys.executable, str(HERE / "script/bench_perf.py"),
        "--device", device, "--dtype", dtype, "--impl", impl,
        "--shape", *[str(dim) for dim in shape],
        "--warmup", str(warmup), "--iters", str(iters), "--json",
    ]
    completed = subprocess.run(
        command, cwd=str(ROOT), env=os.environ.copy(), capture_output=True,
        text=True, check=True, timeout=300,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.lstrip().startswith("{"):
            result = json.loads(line)
            result["case"] = case_name
            return result
    raise RuntimeError(
        f"{case_name}/{dtype}/{impl} 未产生 JSON。\n"
        f"stdout:\n{completed.stdout[-2000:]}\n"
        f"stderr:\n{completed.stderr[-2000:]}"
    )


def _print_markdown(records: list[dict]) -> None:
    by_key = {(r["case"], r["dtype"], r["impl"]): r for r in records}
    print("| case | shape | dtype | Triton ms | FlagGems ms | 原生 ms | "
          "Triton/原生 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for case_name, shape in CASES:
        for dtype in DTYPES:
            tri = by_key[(case_name, dtype, "triton")]["latency_ms"]
            gems = by_key[(case_name, dtype, "flaggems")]["latency_ms"]
            native = by_key[(case_name, dtype, "native")]["latency_ms"]
            shape_text = "×".join(str(dim) for dim in shape)
            print(f"| {case_name} | {shape_text} | {dtype} | {tri:.6f} | "
                  f"{gems:.6f} | {native:.6f} | {tri / native:.3f}× |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="p800-kunlunxin")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", help="可选 JSON 输出路径")
    args = parser.parse_args()

    from common.device import load_profile
    profile = load_profile(args.profile)
    # 默认按官方 profile 暴露物理卡；允许部署环境显式选择同型号健康卡。
    # 例如新服务器物理 XPU 2 的小张量 XMLIR 路径会停住，本次使用 0,1。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", profile.visible_devices)
    os.environ.setdefault(
        "XPU_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    records = []
    total = len(CASES) * len(DTYPES) * len(IMPLS)
    completed = 0
    for case_name, shape in CASES:
        for dtype in DTYPES:
            for impl in IMPLS:
                completed += 1
                print(f"[{completed:02d}/{total}] {case_name} {dtype} {impl}",
                      flush=True)
                records.append(_run_one(
                    profile.torch_device, case_name, shape, dtype, impl,
                    args.warmup, args.iters,
                ))

    print()
    _print_markdown(records)
    by_key = {(r["case"], r["dtype"], r["impl"]): r for r in records}
    ratios = []
    for case_name, _shape in CASES:
        for dtype in DTYPES:
            triton_ms = by_key[(case_name, dtype, "triton")]["latency_ms"]
            native_ms = by_key[(case_name, dtype, "native")]["latency_ms"]
            ratios.append(triton_ms / native_ms)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    bundle = {
        "schema": 1,
        "operator": "aten.sigmoid",
        "profile": profile.name,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "command": (
            "python3 ops/sigmoid/script/bench_matrix.py "
            f"--profile {profile.name} --warmup {args.warmup} "
            f"--iters {args.iters} --out {args.out or '<path>'}"
        ),
        "device": profile.torch_device,
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "profile_visible_devices": profile.visible_devices,
        "environment": {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "triton": _version("triton"),
            "flag_gems": _version("flag-gems"),
        },
        "default_route": "native",
        "verification_route": "triton",
        "isolated_subprocess_per_case": True,
        "summary": {
            "cases": len(records),
            "triton_combinations": len(ratios),
            "triton_faster_than_native": sum(ratio < 1.0 for ratio in ratios),
            "triton_native_ratio_min": round(min(ratios), 6),
            "triton_native_ratio_max": round(max(ratios), 6),
        },
        "records": records,
    }
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"\nJSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
