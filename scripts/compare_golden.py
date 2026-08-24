#!/usr/bin/env python3
"""跨设备黄金输出对比。

用法:
  python3 scripts/compare_golden.py --devices p800-kunlunxin,cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "golden"


def load(device: str) -> dict:
    p = GOLDEN_DIR / f"{device}_golden.json"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少 {device} 黄金输出，请先: "
            f"python3 scripts/build_golden.py --device {device}")
    return json.loads(p.read_text())


def common_prefix_len(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", required=True,
                    help="逗号分隔，如 p800-kunlunxin,cpu,nvidia")
    args = ap.parse_args()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    goldens = {d: load(d) for d in devices}

    print("=" * 72)
    print("黄金输出元信息")
    print("=" * 72)
    for d, g in goldens.items():
        print(f"  {d:18s} engine={g.get('engine', 'vllm'):12s} "
              f"snapshots={len(g['snapshots'])} "
              f"stable_prefix={min(g['stable_prefix_lens'])} "
              f"prompts={len(g['prompts'])} sha={g['prompts_sha256']}")

    # prompts 一致性检查（不一致则对比无意义）
    shas = {d: g["prompts_sha256"] for d, g in goldens.items()}
    if len(set(shas.values())) > 1:
        print("\n⚠️ prompts 哈希不一致，跨设备对比无意义:")
        for d, s in shas.items():
            print(f"  {d}: {s}")
        sys.exit(1)

    n_prompts = min(len(g["prompts"]) for g in goldens.values())
    print("\n" + "=" * 72)
    print(f"跨设备共识前缀一致长度（{n_prompts} prompts）")
    print("=" * 72)
    for i in range(n_prompts):
        row = f"  prompt{i}: "
        for a in devices:
            for b in devices:
                if a >= b:
                    continue
                k = common_prefix_len(
                    goldens[a]["consensus_prefix"][i],
                    goldens[b]["consensus_prefix"][i])
                row += f"{a.split('-')[0]}~{b.split('-')[0]}={k:>2d}  "
        print(row)

    print("\n解读: 同硬件栈升级后长度骤降=数值行为漂移; 跨硬件长度短属")
    print("正常（bf16 舍入差异被随机权重放大），关注同硬件纵向变化。")


if __name__ == "__main__":
    main()
