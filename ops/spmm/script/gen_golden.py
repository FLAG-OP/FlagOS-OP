#!/usr/bin/env python3
"""Generate deterministic CPU golden data for CSR spmm (issue #57)."""
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
        m, k, n = case["shapes"][si]
        seed = spec["seed_base"] + count
        g = torch.Generator().manual_seed(seed)
        dense_a = torch.rand(m, k, generator=g)
        dense_a = (dense_a * (torch.rand(m, k, generator=g)
                              < spec["density"])).to(dtype)
        csr = dense_a.to_sparse_csr()
        b = torch.randn(k, n, generator=g).to(dtype)
        # fp64 独立高精度参考（判卷标准; 见 reference.py 文件头口径）
        expected = ref_mod.spmm_reference(csr, b)
        # 官方对拍: CPU 原生稀疏路径 fp16/bf16 未实现（MKL 限制）——
        # fp32 走 torch.sparse.mm, 其余稠密等价
        if dtype is torch.float32:
            official = torch.sparse.mm(csr, b)
        else:
            official = csr.to_dense() @ b
        rel = ((expected.float() - official.float()).abs()
               / official.float().abs().clamp_min(1.0)).max().item()
        assert rel < 5e-2, "ref vs official " + dtype_name
        name = f"{count:05d}_basic_{dtype_name}_{si}.pt"
        path = out_dir / name
        torch.save({"csr": csr, "dense": b, "expected": expected}, path)
        index["files"].append({
            "file": name, "sha256": _sha(path), "case": "basic",
            "dtype": dtype_name, "shape": [m, k, n],
        })
        count += 1

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print("spmm golden:", count, "cases ->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
