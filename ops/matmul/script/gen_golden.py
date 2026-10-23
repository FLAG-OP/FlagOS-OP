#!/usr/bin/env python3
"""Generate deterministic CPU forward and backward matmul golden cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent.parent))
from ops.matmul.reference import matmul_backward_reference, matmul_reference  # noqa: E402


def _input(shape, generator, layout, dtype, scale):
    shape = tuple(shape)
    if layout == "transpose":
        base = shape[:-2] + (shape[-1], shape[-2])
        return (torch.randn(base, generator=generator).to(dtype) * scale
                ).transpose(-2, -1)
    if layout != "contiguous":
        raise ValueError(f"unknown layout: {layout}")
    return torch.randn(shape, generator=generator).to(dtype) * scale


def make_case(case, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    scale = case.get("scale", 0.25)
    a = _input(case["a_shape"], generator,
               case.get("layout_a", "contiguous"), dtype, scale)
    b = _input(case["b_shape"], generator,
               case.get("layout_b", "contiguous"), dtype, scale)
    if case.get("special") == "zeros_a":
        a.zero_()
    output_shape = torch.matmul(a.float(), b.float()).shape
    grad_output = torch.randn(output_shape, generator=generator).to(dtype)
    return a, b, grad_output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "goldendata")
    args = ap.parse_args()
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = {"op": "matmul", "device": "cpu", "files": []}
    filenames = set()
    count = 0

    for case in spec["cases"]:
        for dtype_name in spec["dtypes"]:
            dtype = getattr(torch, dtype_name)
            for seed_no in range(spec["seeds_per_case"]):
                seed = spec["seed_base"] + count
                a, b, grad_output = make_case(case, dtype, seed)
                expected = matmul_reference(a, b)
                grad_a, grad_b = matmul_backward_reference(a, b, grad_output)

                name = f"{case['name']}_{dtype_name}_s{seed_no}.pt"
                if name in filenames:
                    raise ValueError(f"duplicate golden filename: {name}")
                filenames.add(name)
                path = data_dir / name
                torch.save({
                    "inputs": [a, b], "expected": expected,
                    "grad_output": grad_output,
                    "grad_a": grad_a, "grad_b": grad_b,
                    "dtype": dtype_name, "seed": seed, "case": case["name"],
                }, path)
                index["files"].append({
                    "file": f"data/{name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
                count += 1

    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"[gen-golden] {count} forward/backward cases -> {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
