#!/usr/bin/env python3
"""漂移确认实验: 同一配置独立跑 N 次，量化跨进程非确定性。

用法:
  python3 scripts/drift_study.py --device p800-kunlunxin --runs 8

输出:
  results/drift_<device>_<mode>.json   原始快照
  控制台分析: 逐位置一致率 / 首分歧位置 / 全员一致前缀长度
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_once(profile_name: str, mode: str, idx: int, out_dir: Path) -> dict:
    from common.device import load_profile
    profile = load_profile(profile_name)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = profile.visible_devices
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env.pop("VLLM_FL_PLUGIN_MODULES", None)
    env.pop("VLLM_FL_PER_OP", None)
    if mode == "reference":
        env["VLLM_FL_PER_OP"] = "silu_and_mul=reference"
    # vendor 模式: 不设 PER_OP，走平台默认（本机= vendor.kunlunxin）

    out = out_dir / f"drift_{mode}_{idx:02d}.json"
    cmd = [sys.executable,
           str(ROOT / "tests/framework_level/_vllm_runner.py"),
           str(out), "--device", profile_name, "--route", "b"]
    print(f"[drift] run {idx + 1} mode={mode} ...", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return json.loads(out.read_text())


def analyze(snapshots: list[dict], label: str) -> dict:
    n_runs = len(snapshots)
    n_prompts = len(snapshots[0]["output_token_ids"])
    print(f"\n===== {label}: {n_runs} 次独立运行 =====")

    report = {"runs": n_runs, "prompts": []}
    all_stable = []
    for pi in range(n_prompts):
        seqs = [s["output_token_ids"][pi] for s in snapshots]
        L = min(len(s) for s in seqs)

        # 全员一致前缀长度
        stable = 0
        while stable < L and all(s[stable] == seqs[0][stable] for s in seqs):
            stable += 1
        all_stable.append(stable)

        # 逐位置多数票一致率
        pos_majority = []
        for k in range(min(L, 32)):
            c = Counter(s[k] for s in seqs if k < len(s))
            pos_majority.append(c.most_common(1)[0][1] / n_runs)

        # 两两首分歧位置
        first_div = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                k = 0
                while k < min(len(seqs[i]), len(seqs[j])) and seqs[i][k] == seqs[j][k]:
                    k += 1
                first_div.append(k)

        p8 = sum(1 for d in first_div if d >= 8) / len(first_div)
        print(f"  prompt{pi}: 全员一致前缀={stable:>2d}  "
              f"两两首分歧>=8token比例={p8:.0%}  "
              f"前8位多数票一致率={[f'{x:.0%}' for x in pos_majority[:8]]}")
        report["prompts"].append({
            "index": pi, "stable_prefix_all_runs": stable,
            "pairwise_first_div_min": min(first_div),
            "pairwise_ge8_ratio": round(p8, 3),
            "pos_majority_agree_32": [round(x, 3) for x in pos_majority],
        })

    print(f"  结论: 最短全员一致前缀 = {min(all_stable)} token")
    report["min_stable_prefix_all_runs"] = min(all_stable)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--mode", default="reference", choices=["reference", "vendor"])
    args = ap.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    snapshots = [run_once(args.device, args.mode, i, out_dir)
                 for i in range(args.runs)]
    report = analyze(snapshots, f"{args.device}/{args.mode}")
    report["mode"] = args.mode
    report["device"] = args.device
    (out_dir / f"drift_{args.device}_{args.mode}.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n已保存: results/drift_{args.device}_{args.mode}.json")


if __name__ == "__main__":
    main()
