#!/usr/bin/env python3
"""FlagOS 算子模板库统一矩阵入口。

用法:
  python3 run.py --route a1 --level op --device p800-kunlunxin
  python3 run.py --all --device p800-kunlunxin
  python3 run.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ROUTES = ["a1", "a2", "b"]
LEVELS = ["kernel", "op", "framework"]


def load_test(route: str, level: str):
    if level == "kernel":
        mod = __import__(f"tests.kernel_level.test_{route}", fromlist=["run"])
    elif level == "op":
        mod = __import__(f"tests.op_level.test_{route}", fromlist=["run"])
    else:
        mod = __import__(f"tests.framework_level.test_{route}", fromlist=["run"])
    return mod.run


def run_cell(route: str, level: str, profile_name: str, *,
             record: bool = True) -> bool:
    from common.device import load_profile
    profile = load_profile(profile_name)

    print()
    print("=" * 64)
    print(f"  MATRIX CELL: route={route} level={level} device={profile.name}")
    print(f"  {profile.summary()}")
    print("=" * 64)

    t0 = time.perf_counter()
    try:
        ok = bool(load_test(route, level)(profile))
        err = None
    except Exception:
        ok = False
        err = traceback.format_exc()
        print(err)
    dt = time.perf_counter() - t0

    status = "PASS" if ok else "FAIL"
    print(f"  CELL RESULT: {status} ({dt:.1f}s)")

    if record:
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        out = results_dir / f"{profile.name}_{route}_{level}.json"
        out.write_text(json.dumps({
            "device": profile.name, "route": route, "level": level,
            "status": status, "seconds": round(dt, 1),
            "error": err.splitlines()[-1] if err else None,
        }, indent=2))
    return ok


def main() -> None:
    from common.device import list_profiles

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", choices=ROUTES)
    ap.add_argument("--level", choices=LEVELS)
    ap.add_argument("--device", default=None,
                    help="设备 profile 名（缺省自动探测）")
    ap.add_argument("--all", action="store_true",
                    help="跑全部 9 格矩阵 + 跨层一致性")
    ap.add_argument("--consistency", action="store_true",
                    help="仅跑 算子库层<->框架层 跨层一致性验证")
    ap.add_argument("--list", action="store_true", help="列出 profile 与矩阵")
    args = ap.parse_args()

    if args.list:
        print("可用设备 profile:")
        for n in list_profiles():
            print(f"  {n}")
        print("\n矩阵格:")
        for r in ROUTES:
            for l in LEVELS:
                print(f"  route={r:2s} level={l:9s} -> "
                      f"python3 run.py --route {r} --level {l}")
        return

    if not (args.all or args.consistency or (args.route and args.level)):
        ap.error("需要 --all / --consistency，或同时指定 --route 和 --level")

    if args.device:
        device = args.device
    else:
        from common.device import detect_profile
        device = detect_profile().name
        print(f"auto-detected device profile: {device}")

    if args.all:
        args.consistency = True
    cells = [(r, l) for r in ROUTES for l in LEVELS] if args.all \
        else [(args.route, args.level)] if (args.route and args.level) else []

    passed, failed = [], []
    for r, l in cells:
        ok = run_cell(r, l, device)
        (passed if ok else failed).append((r, l))

    if args.consistency:
        from common.device import load_profile
        from tests.kernel_level import test_consistency
        print()
        print("=" * 64)
        print(f"  CONSISTENCY: 算子库层<->框架层 device={device}")
        print("=" * 64)
        t0 = time.perf_counter()
        try:
            ok = bool(test_consistency.run(load_profile(device)))
            err = None
        except Exception:
            ok = False
            err = traceback.format_exc()
            print(err)
        (passed if ok else failed).append(("consistency", "算子库↔框架"))
        dt = time.perf_counter() - t0
        print(f"  CONSISTENCY RESULT: {'PASS' if ok else 'FAIL'} ({dt:.1f}s)")
        out = ROOT / "results" / f"{device}_consistency.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({
            "device": device, "route": "consistency", "level": "算子库层↔框架层",
            "status": "PASS" if ok else "FAIL",
            "seconds": round(dt, 1),
            "error": err.splitlines()[-1] if err else None,
        }, indent=2))

    print()
    print("=" * 64)
    print(f"MATRIX SUMMARY [{device}]: {len(passed)} passed, {len(failed)} failed")
    for r, l in passed:
        print(f"  PASS  {r} x {l}")
    for r, l in failed:
        print(f"  FAIL  {r} x {l}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
