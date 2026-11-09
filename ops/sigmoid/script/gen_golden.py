#!/usr/bin/env python3
"""根据输入规格生成 sigmoid 黄金数据。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _load_reference():
    spec = importlib.util.spec_from_file_location("sigmoid_reference",
                                                  HERE / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sigmoid_reference


def _inject_special(tensor: torch.Tensor, name: str) -> None:
    flat = tensor.reshape(-1)
    if name == "zeros":
        flat[:min(8, flat.numel())] = 0
    elif name == "large":
        values = torch.tensor([-100, -20, 20, 100], dtype=tensor.dtype)
        flat[:min(4, flat.numel())] = values[:min(4, flat.numel())]
    elif name == "boundary":
        values = torch.tensor([-12, -1, 0, 1, 12], dtype=tensor.dtype)
        flat[:min(5, flat.numel())] = values[:min(5, flat.numel())]
    else:
        raise ValueError(f"unknown special case: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(HERE / "goldendata/inputs_spec.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(HERE / "goldendata"))
    args = parser.parse_args()

    input_spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    reference = _load_reference()
    data_dir = Path(args.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = {"op": input_spec["op"], "device": args.device, "files": []}
    count = 0
    for case in input_spec["cases"]:
        for shape in case["shapes"]:
            for dtype_name in case["dtypes"]:
                dtype = getattr(torch, dtype_name)
                repeats = case.get("seeds_per_case",
                                   input_spec["seeds_per_case"])
                for repeat in range(repeats):
                    seed = input_spec["seed_base"] + count
                    generator = torch.Generator().manual_seed(seed)
                    factory = (torch.randn if case["distribution"] == "normal"
                               else torch.rand)
                    x = factory(*shape, generator=generator).to(dtype)
                    x.mul_(case["scale"])
                    for special in case.get("special", []):
                        _inject_special(x, special)
                    expected = reference(x)
                    shape_tag = "x".join(str(dim) for dim in shape)
                    filename = (f"{case['name']}_{dtype_name}_{shape_tag}_"
                                f"s{repeat}.pt")
                    path = data_dir / filename
                    torch.save({"inputs": [x], "expected": expected,
                                "shape": list(shape), "dtype": dtype_name,
                                "seed": seed, "case": case["name"]}, path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                    index["files"].append({"file": f"data/{filename}",
                                           "sha256_16": digest})
                    count += 1
    (Path(args.out) / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")
    print(f"[gen-golden] {count} 组 -> {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
