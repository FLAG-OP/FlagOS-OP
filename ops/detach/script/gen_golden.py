#!/usr/bin/env python3
"""生成 detach 黄金（alias 类）: 输入 + 元数据/别名期望。"""
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
    return mod.detach_reference


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
                for k in range(spec["seeds_per_case"]):
                    seed = spec["seed_base"] + n
                    g = torch.Generator().manual_seed(seed)
                    x = torch.randn(*shape, generator=g).to(dt)
                    exp = ref_fn(x)
                    name = (f"{case['name']}_{dt_name}_"
                            f"{'x'.join(map(str, shape))}_k{k}.pt")
                    torch.save({
                        "inputs": [x],
                        "expected_meta": {"shape": list(exp.shape),
                                          "dtype": dt_name,
                                          "stride": list(exp.stride()),
                                          "requires_grad": bool(exp.requires_grad),
                                          "shares_storage": True},
                        "dtype": dt_name, "seed": seed, "case": case["name"],
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
