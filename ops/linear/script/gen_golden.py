#!/usr/bin/env python3
"""Generate deterministic CPU golden data for aten::linear (issue #55)."""
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

    ref_mod = _load("linear_reference", "reference.py")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    out_dir = Path(args.out) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pt"):
        old.unlink()
    index = {"op": spec["op"], "device": "cpu", "files": []}
    count = 0

    for case_name in ("basic", "large"):
        case = spec["cases"][case_name]
        for dtype_name, ishape, n_out, bias_on, seed_no in itertools.product(
                case["dtypes"], case["input_shapes"], case["out_features"],
                case["bias"], range(case["seeds_per_case"])):
            dtype = getattr(torch, dtype_name)
            seed = spec["seed_base"] + count
            g = torch.Generator().manual_seed(seed)
            k_in = ishape[-1]
            x = (torch.randn(*ishape, generator=g)
                 * case["input_scale"]).to(dtype)
            w = (torch.randn(n_out, k_in, generator=g)
                 * case["weight_scale"]).to(dtype)
            b = (torch.randn(n_out, generator=g)
                 * case["weight_scale"]).to(dtype) if bias_on else None
            expected = ref_mod.linear_reference(x, w, b)
            # 官方 CPU aten 语义对拍
            official = torch.nn.functional.linear(x, w, b)
            rel = ((expected.float() - official.float()).abs()
                   / official.float().abs().clamp_min(1.0)).max().item()
            assert rel < 5e-2, f"ref vs official {dtype_name}: {rel}"
            name = f"{count:05d}_{case_name}_{dtype_name}_b{int(bias_on)}.pt"
            path = out_dir / name
            torch.save({"x": x, "w": w, "bias": b, "expected": expected},
                       path)
            index["files"].append({
                "file": name, "sha256": _sha(path), "case": case_name,
                "dtype": dtype_name, "input_shape": list(ishape),
                "out_features": n_out, "bias": bias_on,
            })
            count += 1

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print(f"linear golden: {count} cases -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
