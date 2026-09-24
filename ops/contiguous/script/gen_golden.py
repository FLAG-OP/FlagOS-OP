#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 contiguous 黄金输入与参考输出。

每个组合生成 `contig`（别名快路径）与 `view`（transpose，拷贝路径）
两种 layout。

用法:
  python3 script/gen_golden.py --device cpu
"""
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
    return mod.contiguous_reference


SPECIALS = {
    "zeros": lambda t: t.__setitem__(slice(0, 8), 0.0),
    "large": lambda t: t.__setitem__(slice(8, 16), 1e4),
    "boundary": lambda t: t.__setitem__(
        slice(16, 24), torch.finfo(t.dtype).max / 2),
}


def _make(base, layout):
    if layout == "contig":
        return base
    return base.transpose(0, 1)          # 非连续视图


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
                    dist = (torch.randn
                            if case["distribution"] == "normal" else torch.rand)
                    base = dist(*shape, generator=g).to(dt) * case["scale"]
                    for sp in case.get("special", []):
                        SPECIALS[sp](base)
                    for layout in ("contig", "view"):
                        if layout == "view" and base.dim() < 2:
                            continue
                        t = _make(base, layout)
                        expected = ref_fn(t)
                        stag = "x".join(str(s) for s in shape)
                        name = (f"{case['name']}_{dt_name}_{layout}_{stag}"
                                f"_s{k}.pt")
                        torch.save({
                            "inputs": [t],
                            "expected": expected,
                            "shape": list(shape),
                            "dtype": dt_name,
                            "layout": layout,
                            "seed": seed,
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
