#!/usr/bin/env python3
"""运行性能用例并落盘结构化记录（可更新入库基线）。

用法:
  python3 scripts/perf_run.py --device p800-kunlunxin            # 全部用例
  python3 scripts/perf_run.py --device ... --pattern bmm         # 子串过滤
  python3 scripts/perf_run.py --device ... --update-baseline     # 更新 perf/baselines/
  python3 scripts/perf_run.py --list

每个用例在独立子进程中执行（消除用例间分配器/缓存污染，
见 docs/known-issues.md #10）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from common.device import detect_profile, list_profiles, load_profile
    from common.perf import run_case, save_baseline, save_run_record
    from common.perf_registry import load_cases

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="设备 profile（缺省自动探测）")
    ap.add_argument("--group", action="append",
                    help="过滤 group（example/matrix-kernel，可多次）")
    ap.add_argument("--pattern", default=None, help="按 case_id 子串过滤")
    ap.add_argument("--update-baseline", action="store_true",
                    help="把本次成功记录合并进 perf/baselines/<device>.json")
    ap.add_argument("--list", action="store_true", help="列出用例（不执行）")
    ap.add_argument("--_one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.device:
        profile = load_profile(args.device)
    elif args.list:
        profile = load_profile("cpu")  # --list 只需一个 profile 上下文
    else:
        profile = detect_profile()

    cases = load_cases(profile, groups=args.group, pattern=args.pattern)

    # 内部模式: 单用例执行并落盘（由主进程按用例子进程调用）
    if args._one:
        case = next((c for c in cases if c.case_id == args._one), None)
        if case is None:
            print(f"case 不存在: {args._one}")
            return 1
        rec = run_case(case, profile)
        save_run_record(rec)
        print("RESULT " + json.dumps(rec.to_dict()))
        return 0

    if args.list:
        print(f"性能用例（设备上下文: {profile.name}，共 {len(cases)} 个）:")
        for c in cases:
            print(f"  [{c.group:13s}] {c.case_id}"
                  f"  ({c.warmup}+{c.iters})")
        if not cases:
            print("  (无)")
        return 0

    print(f"[perf-run] 设备: {profile.summary()}")
    print(f"[perf-run] 用例: {len(cases)} 个"
          + (f"（过滤: pattern={args.pattern}）" if args.pattern else ""))

    records = []
    n_pass = n_skip = 0
    print(f"\n  {'case_id':52s} {'latency_ms':>10s}  附加指标")
    print("  " + "-" * 76)
    for c in cases:
        r = subprocess.run(
            [sys.executable, __file__, "--_one", c.case_id,
             "--device", profile.name],
            capture_output=True, text=True, cwd=ROOT)
        line = next((l for l in r.stdout.splitlines()
                     if l.startswith("RESULT ")), None)
        if r.returncode != 0 or line is None:
            n_skip += 1
            err = (r.stderr.strip().splitlines() or ["unknown"])[-1]
            print(f"  {c.case_id:52s} {'SKIP':>10s}  {err[:60]}")
            continue
        rec_dict = json.loads(line[7:])
        records.append(rec_dict)
        extra = {k: v for k, v in rec_dict["metrics"].items()
                 if k != "latency_ms"}
        n_pass += 1
        print(f"  {c.case_id:52s} "
              f"{rec_dict['metrics']['latency_ms']:>10.3f}  {extra or ''}")

    print(f"\n[perf-run] 完成: {n_pass} 记录 / {n_skip} 跳过"
          f" → results/perf/runs/{profile.name}/")
    if records:
        print(f"[perf-run] 对比: python3 scripts/perf_compare.py --device {profile.name}")
    if args.update_baseline:
        if not records:
            print("[perf-run] 无成功记录，不更新基线")
            return 1
        from common.perf import save_baseline_dicts
        out = save_baseline_dicts(profile.name, records)
        print(f"[perf-run] 基线已更新: {out}")
        print("[perf-run] 提交基线时请在提交信息说明原因（算子语义变更/环境升级等）")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
