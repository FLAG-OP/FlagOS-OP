#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成 type_as 黄金输入与参考输出。

与通用模板的差异: type_as 是双输入算子（self, other），other 只贡献
dtype。参考输出在 CPU 上由 reference.py 的原生 .to() 计算（cast 无算术
放大，CPU 即为最可信判卷标准）。

用法:
  python3 script/gen_golden.py                    # CPU 参考（默认）
  python3 script/gen_golden.py --device mlu       # 在设备上生成参考

产物: goldendata/data/<case>.pt（inputs+expected）+ index.json（含 sha256）。
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
    return mod.type_as_reference


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
        for shape in case["shapes"]:
            for dt_name in case["dtypes"]:
                for tdt_name in case.get("target_dtypes", case["dtypes"]):
                    if tdt_name == dt_name:
                        continue          # 同 dtype 走别名快路径，另测
                    dt = getattr(torch, dt_name)
                    tdt = getattr(torch, tdt_name)
                    for k in range(spec["seeds_per_case"]):
                        seed = spec["seed_base"] + n
                        g = torch.Generator().manual_seed(seed)
                        dist = (torch.randn
                                if case["distribution"] == "normal"
                                else torch.rand)
                        self_t = (dist(*shape, generator=g).to(dt)
                                  * case["scale"])
                        for sp in case.get("special", []):
                            SPECIALS[sp](self_t)
                        # other 只贡献 dtype，形状无关；给一个小编号张量
                        other = torch.zeros(4, dtype=tdt)
                        expected = ref_fn(self_t, other).cpu()
                        stag = "x".join(str(s) for s in shape)
                        name = (f"{case['name']}_{dt_name}_to_{tdt_name}"
                                f"_{stag}_s{k}.pt")
                        torch.save({
                            "inputs": [self_t.cpu(), other.cpu()],
                            "expected": expected,
                            "shape": list(shape),
                            "dtype": dt_name,
                            "target_dtype": tdt_name,
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
