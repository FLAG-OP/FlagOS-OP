#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 empty 黄金（元数据，无数值）。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

_MF = {
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
}


def _load_reference():
    p = HERE / "reference.py"
    spec = importlib.util.spec_from_file_location("op_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.empty_reference


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
                for mf_name in case["memory_formats"]:
                    for k in range(spec["seeds_per_case"]):
                        exp = ref_fn(tuple(shape), dtype=dt,
                                     memory_format=_MF[mf_name])
                        name = (f"{case['name']}_{dt_name}_{mf_name}"
                                f"_{'x'.join(map(str, shape))}_k{k}.pt")
                        torch.save({
                            "params": {"shape": list(shape),
                                       "dtype": dt_name,
                                       "memory_format": mf_name},
                            "expected_meta": {
                                "shape": list(exp.shape),
                                "dtype": str(exp.dtype).split(".")[-1],
                                "stride": list(exp.stride()),
                            },
                            "case": case["name"],
                        }, data_dir / name)
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
