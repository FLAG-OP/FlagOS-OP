#!/usr/bin/env python3
"""Generate deterministic CPU golden data for _softmax (issue #55)."""
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
    for dtype_name, si, seed_no in itertools.product(
            case["dtypes"], range(len(case["shapes"])),
            range(case["seeds_per_case"])):
        dtype = getattr(torch, dtype_name)
        shape = case["shapes"][si]
        seed = spec["seed_base"] + count
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(*shape, generator=g) * spec["scale"]).to(dtype)
        expected = ref_mod.softmax_prim_reference(x, case["dim"], False)
        official = torch.ops.aten._softmax(x, case['dim'], False)
        tol = 2e-5 if dtype is torch.float32 else 1e-2
        assert (expected.float() - official.float()).abs().max().item() < tol, \
            "ref vs official " + dtype_name
        name = f"{count:05d}_basic_{dtype_name}_{si}.pt"
        path = out_dir / name
        torch.save({"input": x, "dim": case["dim"], "half_to_float": False,
                    "expected": expected}, path)
        index["files"].append({
            "file": name, "sha256": _sha(path), "case": "basic",
            "dtype": dtype_name, "shape": list(shape),
        })
        count += 1


    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print("_softmax golden:", count, "cases ->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
