#!/usr/bin/env python3
"""对比当前性能运行记录与入库基线（回归门禁）。

默认阈值: 比基线慢 20% → WARN，慢 30% → FAIL（退出码 1）。
门禁只作用于 latency_ms；带宽/算力等指标仅展示。

用法:
  python3 scripts/perf_compare.py --device p800-kunlunxin
  python3 scripts/perf_compare.py --device ... --warn 0.2 --fail 0.3
  python3 scripts/perf_compare.py --device ... --out report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.perf import baseline_path, load_baseline, load_run_records


def fmt_delta(d: float | None) -> str:
    if d is None:
        return "—"
    return f"{d:+.1%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--baseline", default=None, help="基线文件（缺省 perf/baselines/<device>.json）")
    ap.add_argument("--warn", type=float, default=0.20)
    ap.add_argument("--fail", type=float, default=0.30)
    ap.add_argument("--out", default=None, help="Markdown 报告落盘路径")
    args = ap.parse_args()

    if args.baseline:
        base_path = Path(args.baseline)
        base = json.loads(base_path.read_text()) if base_path.is_file() else None
    else:
        base_path = baseline_path(args.device)
        base = load_baseline(args.device)

    runs = load_run_records(args.device)
    if not runs:
        print(f"[perf-compare] 无运行记录: results/perf/runs/{args.device}/")
        print("先执行: python3 scripts/perf_run.py --device " + args.device)
        return 2

    lines = [
        f"# 性能回归对比: {args.device}",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- 阈值: WARN ≥ {args.warn:.0%} · FAIL ≥ {args.fail:.0%}（仅对 latency_ms 门禁）",
    ]
    if base is None:
        lines.append(f"- 基线: **不存在**（{base_path}）→ 全部标 NEW，不判失败")
    else:
        env_b = base.get("env_fingerprint", {})
        env_r = runs[0].get("env_fingerprint", {})
        env_diff = {k: (env_b.get(k), env_r.get(k))
                    for k in set(env_b) | set(env_r)
                    if env_b.get(k) != env_r.get(k)}
        lines += [
            f"- 基线: {base_path}（commit `{str(base.get('git_commit'))[:12]}`，"
            f"更新于 {str(base.get('updated_at'))[:19]}）",
            f"- 环境: {'一致' if not env_diff else '存在差异 ' + json.dumps(env_diff, ensure_ascii=False)}"
            "（仅提示，不阻断）",
        ]

    lines += ["", "| case | 基线 ms | 本次 ms | Δ | 判定 | 附加指标 |", "|---|---|---|---|---|---|"]
    n_fail = n_warn = n_new = 0
    for r in sorted(runs, key=lambda x: x["case_id"]):
        cid = r["case_id"]
        cur = r["metrics"].get("latency_ms")
        bcase = (base or {}).get("cases", {}).get(cid)
        extra = {k: v for k, v in r["metrics"].items() if k != "latency_ms"}
        extra_s = " ".join(f"{k}={v}" for k, v in extra.items())
        if bcase is None:
            status = "NEW"
            n_new += 1
            lines.append(f"| {cid} | — | {cur:.3f} | — | {status} | {extra_s} |")
            continue
        old = bcase["metrics"].get("latency_ms")
        delta = (cur - old) / old if old else None
        if delta is None or delta <= args.warn:
            status = "OK"
        elif delta <= args.fail:
            status = "WARN"
            n_warn += 1
        else:
            status = "FAIL"
            n_fail += 1
        lines.append(f"| {cid} | {old:.3f} | {cur:.3f} | "
                     f"{fmt_delta(delta)} | **{status}** | {extra_s} |")

    base_ids = set((base or {}).get("cases", {}))
    run_ids = {r["case_id"] for r in runs}
    missing = sorted(base_ids - run_ids)
    if missing:
        lines += ["", "**本次未运行的基线用例**（不判失败）: " + ", ".join(missing)]

    verdict = "FAIL" if n_fail else ("WARN" if n_warn else "OK")
    lines += [
        "",
        f"## 结论: {verdict}",
        f"FAIL {n_fail} · WARN {n_warn} · NEW {n_new} · 共 {len(runs)} 条",
    ]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[perf-compare] 报告已写入 {args.out}")
    print(text)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
