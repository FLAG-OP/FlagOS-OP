#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 dropout 黄金。

- deterministic 用例存 expected（原生 dropout，位级）；
- stochastic 用例只存输入与 (p, train)，标记 stochastic=true，
  由 check_accuracy 做结构/统计判定（RNG 流与原生不同）。

用法:
  python3 script/gen_golden.py --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]


def _load_reference():
    p = HERE / "reference.py"
    spec = importlib.util.spec_from_file_location("op_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dropout_reference


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(HERE / "goldendata/inputs_spec.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    ref_fn = _load_reference()
    data_dir = Path(args.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    index = {"op": spec["op"], "device": args.device, "files": []}
    n = 0
    for case in spec["cases"]:
        for shape in case["shapes"]:
            for dt_name in case["dtypes"]:
                dt = getattr(torch, dt_name)
                for setting in case["settings"]:
                    p, train = setting["p"], setting["train"]
                    for k in range(spec["seeds_per_case"]):
                        seed = spec["seed_base"] + n
                        g = torch.Generator().manual_seed(seed)
                        t = (torch.randn(*shape, generator=g).to(dt)
                             * case["scale"])
                        stochastic = (train and 0.0 < p < 1.0)
                        rec = {
                            "inputs": [t],
                            "shape": list(shape),
                            "dtype": dt_name,
                            "p": p, "train": train,
                            "stochastic": stochastic,
                            "seed": seed,
                            "case": case["name"],
                        }
                        if not stochastic:
                            rec["expected"] = ref_fn(t, p, train)
                        stag = "x".join(str(s) for s in shape)
                        name = (f"{case['name']}_{dt_name}_p{p}_{train}"
                                f"_{stag}_k{k}.pt")
                        torch.save(rec, data_dir / name)
                        sha = hashlib.sha256(
                            (data_dir / name).read_bytes()).hexdigest()[:16]
                        index["files"].append({
                            "file": f"data/{name}", "sha256_16": sha})
                        n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[gen-golden] {n} 组 → {data_dir}（index.json 已写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
