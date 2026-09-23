#!/usr/bin/env python3
"""Generate deterministic CPU golden data for aten::embedding."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _index_tensor(spec, shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, int(spec["num_weights"][0]), tuple(shape), generator=g
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()
    assert args.device == "cpu", "golden authority is CPU"

    ref_mod = _load("embedding_reference", "reference.py")
    spec = yaml.safe_load(
        (HERE / "goldendata/inputs_spec.yaml").read_text()
    )
    out_dir = Path(args.out) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"op": spec["op"], "device": "cpu", "files": []}
    count = 0

    basic = spec["cases"]["basic"]
    for dtype_name, dim, shape, pad, seed_no in itertools.product(
        basic["dtypes"], basic["dims"], basic["index_shapes"],
        basic["padding_idx"], range(basic["seeds_per_case"]),
    ):
        dtype = getattr(torch, dtype_name)
        seed = spec["seed_base"] + count
        g = torch.Generator().manual_seed(seed)
        weight = (
            torch.randn(
                int(basic["num_weights"][0]), dim, generator=g
            ) * 0.1
        )
        weight = weight.to(dtype)
        indices = _index_tensor(basic, shape, seed + 1)
        expected = ref_mod.embedding_reference(weight, indices, pad)
        # Cross-check with official CPU aten semantics.
        official = torch.nn.functional.embedding(
            indices, weight, padding_idx=pad
        )
        assert torch.equal(expected, official)
        count += _save(
            out_dir, index, count, dtype_name, weight, indices, expected,
            "basic", pad,
        )

    duplicate = spec["cases"]["duplicates"]
    for dtype_name, pad, seed_no in itertools.product(
        duplicate["dtypes"], duplicate["padding_idx"],
        range(duplicate["seeds_per_case"]),
    ):
        dtype = getattr(torch, dtype_name)
        seed = spec["seed_base"] + 1000 + count
        g = torch.Generator().manual_seed(seed)
        weight = (torch.randn(64, 16, generator=g) * 0.1).to(dtype)
        indices = torch.tensor(duplicate["indices"][0], dtype=torch.long)
        expected = ref_mod.embedding_reference(weight, indices, pad)
        count += _save(
            out_dir, index, count, dtype_name, weight, indices, expected,
            "duplicates", pad,
        )

    empty = spec["cases"]["empty"]
    for dtype_name in empty["dtypes"]:
        dtype = getattr(torch, dtype_name)
        seed = spec["seed_base"] + 2000 + count
        g = torch.Generator().manual_seed(seed)
        weight = (torch.randn(64, 16, generator=g) * 0.1).to(dtype)
        indices = torch.empty((0,), dtype=torch.long)
        expected = ref_mod.embedding_reference(weight, indices, -1)
        count += _save(
            out_dir, index, count, dtype_name, weight, indices, expected,
            "empty", -1,
        )

    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"generated {count} golden cases -> {out_dir}")
    return 0


def _save(out_dir, index, count, dtype_name, weight, indices, expected,
          case, padding_idx):
    name = f"{case}_{dtype_name}_s{count}.pt"
    torch.save(
        {
            "inputs": [weight, indices],
            "expected": expected,
            "dtype": dtype_name,
            "padding_idx": padding_idx,
            "case": case,
        },
        out_dir / name,
    )
    digest = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()[:16]
    index["files"].append({"file": f"data/{name}", "sha256_16": digest})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
