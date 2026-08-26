#!/usr/bin/env python3
"""汇总 3×3 矩阵 + consistency 结果为 Markdown 报告。

用法:
  python3 scripts/report.py --device p800-kunlunxin            # 读 results/
  python3 scripts/report.py --device ... --out report.md       # 落盘
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

ROUTES = ["a1", "a2", "b"]
LEVELS = ["kernel", "op", "framework"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--out", default=None, help="输出 md 路径（缺省打印）")
    args = ap.parse_args()

    cells = {}
    for r in ROUTES:
        for l in LEVELS:
            p = RESULTS / f"{args.device}_{r}_{l}.json"
            cells[(r, l)] = json.loads(p.read_text()) if p.exists() else None
    cons = RESULTS / f"{args.device}_consistency.json"

    lines = [
        f"# 矩阵报告: {args.device}",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 3×3 矩阵",
        "",
        "| 路线 | kernel 层 | op 层 | framework 层 |",
        "|---|---|---|---|",
    ]
    n_pass = n_fail = n_missing = 0
    for r in ROUTES:
        row = f"| **{r.upper()}** |"
        for l in LEVELS:
            c = cells[(r, l)]
            if c is None:
                row += " ⬜ 未跑 |"
                n_missing += 1
            elif c["status"] == "PASS":
                row += f" ✅ {c['seconds']:.0f}s |"
                n_pass += 1
            else:
                row += f" ❌ {c.get('error') or ''} |"
                n_fail += 1
        lines.append(row)

    lines += ["", f"通过 {n_pass} / 失败 {n_fail} / 未跑 {n_missing}", ""]

    if cons.exists():
        c = json.loads(cons.read_text())
        lines += [f"## 跨层一致性: {c['status']} ({c['seconds']}s)", ""]

    lines += [
        "## 解读",
        "- kernel 层: 硬件语言/Triton kernel 直测（精度+哨兵+性能）",
        "- op 层: 注册/分发/拦截验证",
        "- framework 层: 真实 vLLM 推理注入验证",
        "- consistency: 同算子 算子库层直调 vs 框架层 dispatch 张量级一致",
    ]

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"已写入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
