#!/usr/bin/env python3
"""Generate deterministic CPU golden data for layer_norm (issue #55)."""
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()
    assert args.device == "cpu", "golden authority is CPU"

    ref_mod = _load("ref_mod", "reference.py")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    out_dir = Path(args.out) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pt"):
        old.unlink()
    index = {"op": spec["op"], "device": "cpu", "files": []}
    count = 0

    case = spec["cases"]["basic"]
    for dtype_name, ishape, seed_no in itertools.product(
            case["dtypes"], case["input_shapes"],
            range(case["seeds_per_case"])):
        dtype = getattr(torch, dtype_name)
        seed = spec["seed_base"] + count
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(*ishape, generator=g).to(dtype)
        nshape = (ishape[-1],)
        w = (torch.randn(nshape[-1], generator=g) * case["weight_scale"]).to(dtype)
        b = (torch.randn(nshape[-1], generator=g) * case["weight_scale"]).to(dtype)
        expected = ref_mod.layer_norm_reference(x, nshape, w, b, case["eps"])
        official = torch.nn.functional.layer_norm(x, nshape, w, b, case['eps'])
        assert (expected.float() - official.float()).abs().max().item() \
            < (2e-5 if dtype is torch.float32 else 1e-2), \
            "ref vs official " + dtype_name
        name = f"{count:05d}_basic_{dtype_name}.pt"
        path = out_dir / name
        torch.save({"input": x, "weight": w, "bias": b,
                    "normalized_shape": list(nshape), "eps": case["eps"],
                    "expected": expected}, path)
        index["files"].append({
            "file": name, "sha256": _sha(path), "case": "basic",
            "dtype": dtype_name, "shape": list(ishape),
        })
        count += 1

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print(f"layer_norm golden:", count, "cases ->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
