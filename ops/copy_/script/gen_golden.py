#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 copy_ 黄金输入与参考输出（原地）。

dst 提供输出形状/dtype，src 可广播。参考: `dst.copy_(src)`（CPU），
保存 dst 初值与 src，expected = copy 之后的值。

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
    return mod.copy_reference


SPECIALS = {
    "zeros": lambda t: t.__setitem__(slice(0, 8), 0.0),
    "large": lambda t: t.__setitem__(slice(8, 16), 1e4),
    "boundary": lambda t: t.__setitem__(
        slice(16, 24), torch.finfo(t.dtype).max / 2),
}


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
        for dst_shape, src_shape in case["shape_pairs"]:
            for dd_name, sd_name in case["dtype_pairs"]:
                dd = getattr(torch, dd_name)
                sd = getattr(torch, sd_name)
                for k in range(spec["seeds_per_case"]):
                    seed = spec["seed_base"] + n
                    g = torch.Generator().manual_seed(seed)
                    dist = (torch.randn
                            if case["distribution"] == "normal" else torch.rand)
                    dst = (dist(*dst_shape, generator=g).to(dd) * case["scale"])
                    src = (dist(*src_shape, generator=g).to(sd) * case["scale"])
                    for sp in case.get("special", []):
                        SPECIALS[sp](src)          # 特殊值放在被拷入的 src 上
                    expected = dst.clone()
                    ref_fn(expected, src)
                    dtag = "x".join(str(s) for s in dst_shape)
                    stag = "x".join(str(s) for s in src_shape)
                    name = (f"{case['name']}_{dd_name}_{sd_name}"
                            f"_d{dtag}_s{stag}_k{k}.pt")
                    torch.save({
                        "inputs": [dst, src],
                        "expected": expected,
                        "dst_shape": list(dst_shape),
                        "src_shape": list(src_shape),
                        "dtype": dd_name,
                        "src_dtype": sd_name,
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
