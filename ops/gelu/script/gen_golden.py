#!/usr/bin/env python3
"""Generate deterministic CPU golden data for aten::gelu (issue #55)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
_NONFINITE = {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(out_dir: Path, index: dict, count: int, tag: str, dtype_name: str,
          x: torch.Tensor, expected: torch.Tensor, mode: str) -> int:
    name = f"{count:05d}_{tag}_{mode}_{dtype_name}.pt"
    path = out_dir / name
    torch.save({"x": x, "expected": expected, "mode": mode}, path)
    index["files"].append({
        "file": name, "sha256": _sha(path), "tag": tag, "mode": mode,
        "dtype": dtype_name, "shape": list(x.shape),
    })
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()
    assert args.device == "cpu", "golden authority is CPU"

    ref_mod = _load("gelu_reference", "reference.py")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    out_dir = Path(args.out) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.pt"):
        old.unlink()          # 生成物可复现; 清陈旧文件对齐当前 spec
    index = {"op": spec["op"], "device": "cpu", "files": []}
    count = 0

    for mode in spec["modes"]:
        # ── basic: 模式 × dtype × shape × seed ──
        basic = spec["cases"]["basic"]
        for dtype_name, shape, seed_no in __import__("itertools").product(
                basic["dtypes"], basic["shapes"], range(basic["seeds_per_case"])):
            dtype = getattr(torch, dtype_name)
            seed = spec["seed_base"] + count
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn(*shape, generator=g) * basic["scale"]).to(dtype)
            expected = ref_mod.gelu_reference(x, mode)
            # 与官方 CPU aten 语义对拍（三方判卷的权威侧）
            official = torch.nn.functional.gelu(x, approximate=mode)
            err = (expected.float() - official.float()).abs().max().item()
            tol = 1e-5 if dtype is torch.float32 else 1e-2
            assert err < tol, f"reference vs official {mode} {dtype_name}: {err}"
            count += _save(out_dir, index, count, "basic", dtype_name,
                           x, expected, mode)

        # ── boundary: 特殊值 + 非有限位级 parity ──
        bd = spec["cases"]["boundary"]
        for dtype_name in bd["dtypes"]:
            dtype = getattr(torch, dtype_name)
            for row in bd["special_values"]:
                x = torch.tensor([row], dtype=dtype)
                expected = ref_mod.gelu_reference(x, mode)
                count += _save(out_dir, index, count, "special", dtype_name,
                               x, expected, mode)
            # 非有限值: parity case 取 [nan, -inf]（gelu(-inf)=-inf·0=NaN,
            # ref 与官方一致）; ⚠️ +inf 在官方 CPU none 模式=NaN、tanh=+Inf,
            # 而设备侧 erf 公式=+Inf——不定式的两种合法 IEEE 解释, 不进
            # 硬判卷, 单独 INFO case 存官方值（边界记录见 REPORT/PLATFORM）
            vals = [_NONFINITE[k] for k in ("nan", "-inf")]
            x = torch.tensor([vals], dtype=torch.float32).to(dtype)
            expected = ref_mod.gelu_reference(x, mode)
            official = torch.nn.functional.gelu(x, approximate=mode)
            assert torch.equal(torch.isnan(expected), torch.isnan(official)), \
                f"{mode} nan parity 失败（vs official）"
            count += _save(out_dir, index, count, "nonfinite", dtype_name,
                           x, expected, mode)
            x_inf = torch.tensor([[_NONFINITE["inf"]]],
                                 dtype=torch.float32).to(dtype)
            count += _save(out_dir, index, count, "inf_info", dtype_name,
                           x_inf, torch.nn.functional.gelu(x_inf, approximate=mode),
                           mode)

        # ── backward: grad × x 联合生成 ──
        bw = spec["cases"]["backward"]
        for dtype_name, shape, seed_no in __import__("itertools").product(
                bw["dtypes"], bw["shapes"], range(bw["seeds_per_case"])):
            dtype = getattr(torch, dtype_name)
            seed = spec["seed_base"] + count
            g = torch.Generator().manual_seed(seed)
            x = (torch.randn(*shape, generator=g) * bw["scale"]).to(dtype)
            grad = torch.randn(*shape, generator=g).to(dtype)
            expected = ref_mod.gelu_backward_reference(grad, x, mode)
            name = f"{count:05d}_bwd_{mode}_{dtype_name}.pt"
            path = out_dir / name
            torch.save({"x": x, "grad": grad, "expected": expected,
                        "mode": mode}, path)
            index["files"].append({
                "file": name, "sha256": _sha(path), "tag": "backward",
                "mode": mode, "dtype": dtype_name, "shape": list(x.shape),
            })
            count += 1

    (out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print(f"gelu golden: {count} cases -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
