#!/usr/bin/env python3
"""生成 result_type 黄金: dtype 对 + 期望公共 dtype。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(HERE / "goldendata/inputs_spec.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.spec).read_text())
    data_dir = Path(args.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = {"op": spec["op"], "device": args.device, "files": []}
    n = 0
    for case in spec["cases"]:
        for pair in case["dtype_pairs"]:
            a = torch.empty(2, dtype=getattr(torch, pair[0]))
            b = torch.empty(2, dtype=getattr(torch, pair[1]))
            exp = str(torch.result_type(a, b)).split(".")[-1]
            name = f"{case['name']}_{pair[0]}__{pair[1]}.pt"
            torch.save({"dtype_pair": pair, "expected": exp},
                       data_dir / name)
            sha = hashlib.sha256(
                (data_dir / name).read_bytes()).hexdigest()[:16]
            index["files"].append({"file": f"data/{name}", "sha256_16": sha})
            n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[gen-golden] {n} 组 → {data_dir}（index.json 已写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
