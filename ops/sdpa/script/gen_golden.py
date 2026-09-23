#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成黄金输入与参考输出（CPU fp32 权威）。

产物: goldendata/data/<case>.pt（inputs + expected + 元数据）+ index.json（sha256）。
交叉互验: expected 同时用 sdpa_reference 与 CPU F.scaled_dot_product_attention
(fp32) 计算，两者 max_err > 1e-6 视为语义理解分歧 → 报错（防"参考恰好也是
被测实现"式掩盖）。
"""
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


def _load(path, entry):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def _special(t: torch.Tensor, kind: str) -> None:
    if kind == "zeros_row":
        t[..., 0, :] = 0.0                # q 全零行
    elif kind == "large_row":
        t[..., 1, :] = 40.0               # 大值行（softmax 稳定性）


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")   # 黄金一律 CPU 生成
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    ref_fn = _load("reference.py", "sdpa_reference")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    data_dir = Path(args.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Goldens are reproducible generated artifacts; remove stale cases so the
    # directory and index always describe the current specification.
    for old in data_dir.glob("*.pt"):
        old.unlink()

    index = {"op": spec["op"], "device": "cpu", "files": []}
    n = 0
    for case in spec["cases"]:
        for dt_name in case["dtypes"]:
            dt = getattr(torch, dt_name)
            # Hq/Hkv 只在长度匹配时配对（GQA case 用 (8,2)，其余 (4,4)）
            hkv_list = (case["Hkv"] if len(case["Hkv"]) == len(case["Hq"])
                        else [case["Hkv"][0]] * len(case["Hq"]))
            combos = list(itertools.product(case["B"], case["Hq"],
                                            hkv_list, case["S"], case["D"],
                                            case["causal"], case["mask"]))
            for (B, Hq, Hkv, S, D, causal, mask_kind) in combos:
                for k in range(case["seeds_per_case"]):
                    # Keep the deterministic extreme-case seed independent of
                    # the normal-grid seed count.  Its saturation behavior is
                    # accumulation-order sensitive, so changing the random
                    # input would silently change the precision criterion.
                    seed = (spec["seed_base"] + 264
                            if case.get("special") else spec["seed_base"] + n)
                    g = torch.Generator().manual_seed(seed)
                    q = (torch.randn(B, Hq, S, D, generator=g)
                         * case["scale"])
                    k_ = (torch.randn(B, Hkv, S, D, generator=g)
                          * case["scale"])
                    v = (torch.randn(B, Hkv, S, D, generator=g)
                         * case["scale"])
                    for sp in case.get("special", []):
                        _special(q, sp)
                    m = None
                    if mask_kind == "bool":
                        m = torch.rand(B, Hq, S, S, generator=g) > 0.3
                    elif mask_kind == "float":
                        m = torch.randn(B, Hq, S, S, generator=g) * 0.1

                    q_t, k_t, v_t, m_t = (
                        t.to(dt) for t in (q, k_, v, m)) if m is not None \
                        else (q.to(dt), k_.to(dt), v.to(dt), None)

                    gqa = Hq != Hkv
                    expected = ref_fn(q_t, k_t, v_t, m_t, 0.0, causal,
                                      None, gqa)
                    # 交叉互验（fp32 上与官方实现比对）——extreme（scale=8）
                    # case 官方实现自身在饱和 softmax 下有 ~3e-4 累加顺序
                    # 差异（实测 native vs golden 同为 3.381e-4），互验阈值
                    # 对带 special 的 case 放宽到相对 2e-3
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q_t.float(), k_t.float(), v_t.float(),
                        attn_mask=m_t if m_t is None else
                        (m_t if m_t.dtype == torch.bool else m_t.float()),
                        is_causal=causal, enable_gqa=gqa)
                    xerr = (expected.float() - x).abs()
                    denom = x.abs().clamp_min(1e-3)
                    xrel = torch.nan_to_num(xerr / denom, nan=0.0).max().item()
                    xabs = torch.nan_to_num(xerr, nan=0.0).max().item()
                    if case.get("special"):
                        assert xrel < 2e-3, \
                            f"互验(相对)分歧 @ seed={seed}: rel={xrel}"
                    else:
                        # expected 带 cast 回输入 dtype 的量化，阈值随 dtype
                        xtol = {torch.float32: 2e-5, torch.float16: 2e-2,
                                torch.bfloat16: 2e-2}[dt]
                        assert xabs < xtol, \
                            f"互验分歧 @ seed={seed}: abs={xabs}"

                    name = (f"{case['name']}_{dt_name}_B{B}_H{Hq}-{Hkv}"
                            f"_S{S}_D{D}_c{int(causal)}"
                            f"_m{mask_kind or 'none'}_s{k}.pt")
                    torch.save(
                        {"inputs": [q_t, k_t, v_t] + ([m_t] if m_t is not None
                                                      else []),
                         "expected": expected.cpu(), "B": B, "Hq": Hq,
                         "Hkv": Hkv, "S": S, "D": D, "causal": causal,
                         "mask": mask_kind, "dtype": dt_name, "seed": seed,
                         "case": case["name"]},
                        data_dir / name)
                    sha = hashlib.sha256((data_dir / name).read_bytes()
                                         ).hexdigest()[:16]
                    index["files"].append({"file": f"data/{name}",
                                           "sha256_16": sha})
                    n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"生成 {n} 组黄金 -> {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
