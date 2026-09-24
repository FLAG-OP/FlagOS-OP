#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 empty_like 黄金（元数据，无数值）。

存 self 的元数据 + 参考输出的 (shape, dtype, stride)；不含数值。
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

_MF = {
    "preserve_format": torch.preserve_format,
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
}


def _load_reference():
    p = HERE / "reference.py"
    spec = importlib.util.spec_from_file_location("op_reference", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.empty_like_reference


def _make_view(base, view):
    if view == "contig":
        return base
    if view == "transpose":
        return base.transpose(0, 1)
    if view == "channels_last":
        return base.to(memory_format=torch.channels_last)
    raise ValueError(view)


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
                    for view in case["views"]:
                        if view == "channels_last" and len(shape) != 4:
                            continue
                        for k in range(spec["seeds_per_case"]):
                            seed = spec["seed_base"] + n
                            g = torch.Generator().manual_seed(seed)
                            base = torch.randn(*shape, generator=g).to(dt)
                            self_t = _make_view(base, view)
                            exp = ref_fn(self_t, memory_format=_MF[mf_name])
                            name = (f"{case['name']}_{dt_name}_{mf_name}"
                                    f"_{view}_{'x'.join(map(str, shape))}"
                                    f"_k{k}.pt")
                            torch.save({
                                "inputs": [self_t],
                                "expected_meta": {
                                    "shape": list(exp.shape),
                                    "dtype": str(exp.dtype).split(".")[-1],
                                    "stride": list(exp.stride()),
                                },
                                "memory_format": mf_name,
                                "view": view,
                                "dtype": dt_name,
                                "seed": seed,
                                "case": case["name"],
                            }, data_dir / name)
                            sha = hashlib.sha256(
                                (data_dir / name).read_bytes()
                            ).hexdigest()[:16]
                            index["files"].append({
                                "file": f"data/{name}", "sha256_16": sha})
                            n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[gen-golden] {n} 组 → {data_dir}（index.json 已写）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
