#!/usr/bin/env python3
"""生成 _local_scalar_dense 黄金（host 标量类）: 输入 + 期望标量。"""
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
        for shape in case["shapes"]:
            for dt_name in case["dtypes"]:
                dt = getattr(torch, dt_name)
                for k in range(spec["seeds_per_case"]):
                    seed = spec["seed_base"] + n
                    g = torch.Generator().manual_seed(seed)
                    if dt is torch.bool:
                        val = bool(k % 2)
                    elif dt.is_floating_point:
                        val = round(float(torch.randn(1, generator=g)) * 2, 4)
                    else:
                        val = int(k) - 1
                    t = torch.full(tuple(shape), val, dtype=dt)
                    name = (f"{case['name']}_{dt_name}_"
                            f"{'x'.join(map(str, shape)) or '0d'}_k{k}.pt")
                    torch.save({
                        "inputs": [t],
                        "expected": t.item(),
                        "dtype": dt_name,
                        "shape": list(shape),
                        "case": case["name"],
                    }, data_dir / name)
                    sha = hashlib.sha256(
                        (data_dir / name).read_bytes()).hexdigest()[:16]
                    index["files"].append(
                        {"file": f"data/{name}", "sha256_16": sha})
                    n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[gen-golden] {n} 组 → {data_dir}（index.json 已写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
