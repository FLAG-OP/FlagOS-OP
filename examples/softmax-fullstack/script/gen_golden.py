#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 softmax 黄金输入与参考输出。"""
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
    spec = importlib.util.spec_from_file_location("softmax_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.softmax_reference


SPECIALS = {
    "zeros": lambda t: t.__setitem__(slice(0, 8), 0.0),
    "large": lambda t: t.__setitem__(slice(8, 16), 1e4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    ref = _load_reference()
    data_dir = HERE / "goldendata/data"
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
                    dist = torch.randn if case["distribution"] == "normal" \
                        else torch.rand
                    x = dist(*shape, generator=g).to(dt) * case["scale"]
                    for sp in case.get("special", []):
                        SPECIALS[sp](x)
                    x = x.to(args.device)
                    expected = ref(x).cpu()
                    name = f"{case['name']}_{dt_name}_s{k}.pt"
                    torch.save({"inputs": [x.cpu()], "expected": expected,
                                "shape": shape, "dtype": dt_name,
                                "seed": seed, "case": case["name"]},
                               data_dir / name)
                    sha = hashlib.sha256((data_dir / name).read_bytes()
                                         ).hexdigest()[:16]
                    index["files"].append({"file": f"data/{name}",
                                           "sha256_16": sha})
                    n += 1
    (HERE / "goldendata/index.json").write_text(json.dumps(index, indent=2))
    print(f"[gen-golden] {n} 组 → {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
