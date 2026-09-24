#!/usr/bin/env python3
"""dropout 精度/正确性测试。

- deterministic 黄金（train=False / p=0 / p=1）: 位级判定 + 别名语义
- stochastic（0<p<1）: 无逐元素黄金，按结构 + 统计判定
  （shape/dtype/stride、值域 {0, x/(1-p)}、drop 比例、非别名、seed 可控）

用法:
  python3 script/check_accuracy.py --impl reference --device cpu
  python3 script/check_accuracy.py --impl triton --device mlu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parents[1]

IMPLS = {
    "reference": ("reference.py", "dropout_reference"),
    "torch": ("kernel/torch_level.py", "dropout_torch"),
    "triton": ("kernel/triton_level.py", "dropout_triton"),
}


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def _check_deterministic(fn, d, dev, tol):
    inp = d["inputs"][0].to(dev)
    out = fn(inp, d["p"], d["train"])
    exp = d["expected"].to(dev)
    if d["p"] == 1.0:
        ok = bool(torch.equal(out, exp))
        return ok, "全零" if ok else "p=1 输出非零"
    floor = torch.finfo(inp.dtype).tiny if inp.dtype.is_floating_point else 0
    eq = (out == inp) | ((out - inp).abs() <= tol)
    alias = out is inp
    ok = bool(eq.all().item()) and alias
    return ok, ("别名+值一致" if ok else "确定性分支不一致")


def _native_stride(inp, p, train, dev):
    torch.manual_seed(0)
    return F.dropout(inp, p, train).stride()


def _check_stochastic(fn, d, dev, tol, ratio_tol):
    inp = d["inputs"][0].to(dev)
    p = d["p"]
    scale = 1.0 / (1.0 - p)
    seed = d["seed"]
    reasons = []

    torch.manual_seed(seed)
    out = fn(inp, p, True)
    if torch.is_tensor(out) is False:
        return False, "输出非张量"
    if out.shape != inp.shape or out.dtype != inp.dtype:
        reasons.append("shape/dtype 不符")
    if out.stride() != _native_stride(inp, p, True, dev):
        reasons.append("stride 与原生不符")
    if out.data_ptr() == inp.data_ptr():
        reasons.append("输出与输入别名")

    # 值域: 每个元素要么 0，要么 input*scale
    keep = out != 0
    if keep.any():
        got = out[keep].float()
        want = inp[keep].float() * scale
        rel = ((got - want).abs() / (want.abs() + 1e-6)).max().item()
        if rel > 0.05:
            reasons.append(f"存活元素未按 x/(1-p) 缩放 (rel={rel:.3f})")
    drop_frac = (~keep).float().mean().item()
    if abs(drop_frac - p) > ratio_tol:
        reasons.append(f"drop 比例 {drop_frac:.4f} 偏离 p={p}")

    # seed 可控: 同 seed 确定性 + 不同 seed 敏感
    torch.manual_seed(seed)
    o1 = fn(inp, p, True)
    torch.manual_seed(seed)
    o2 = fn(inp, p, True)
    if not torch.equal(o1, o2):
        reasons.append("同 seed 不确定")
    torch.manual_seed(seed + 1)
    o3 = fn(inp, p, True)
    if torch.equal(o1, o3):
        reasons.append("不同 seed 输出相同（未消费 RNG）")

    return (not reasons), (f"drop={drop_frac:.4f} rel={rel if keep.any() else 0:.3f}"
                           if not reasons else "; ".join(reasons))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", default="reference",
                    help="reference | torch | triton | <相对路径>:<入口函数>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--golden", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    path, entry = args.impl.split(":") if ":" in args.impl else IMPLS[args.impl]
    fn = _load(path, entry)

    gspec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    tol = gspec["tolerance"]["pointwise_abs"]
    ratio_tol = gspec["drop_ratio_tol"]
    idx = json.loads((Path(args.golden) / "index.json").read_text())

    n_pass = n_fail = 0
    for f in idx["files"]:
        d = torch.load(Path(args.golden) / f["file"])
        if d["stochastic"]:
            ok, detail = _check_stochastic(
                fn, d, args.device, tol[d["dtype"]], ratio_tol)
        else:
            ok, detail = _check_deterministic(
                fn, d, args.device, tol[d["dtype"]])
        n_pass, n_fail = n_pass + ok, n_fail + (not ok)
        if not ok:
            print(f"✗ {f['file']}: {detail}")
    print(f"结论: {n_pass} pass / {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
