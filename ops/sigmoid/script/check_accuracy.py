#!/usr/bin/env python3
"""读取黄金数据并检查 sigmoid 候选实现精度。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _load(path: str, entry: str):
    spec = importlib.util.spec_from_file_location(path.replace("/", "_")[:-3],
                                                  HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, entry)


def _resolve_device(name: str) -> str:
    if name == "cpu" or name.startswith(("cuda", "xpu", "npu", "mlu")):
        return name
    from common.device import load_profile
    return load_profile(name).torch_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--golden", default=str(HERE / "goldendata"))
    args = parser.parse_args()
    if args.impl == "reference":
        from ops.sigmoid.reference import sigmoid_reference
        function = sigmoid_reference
    elif args.impl == "torch":
        from ops.sigmoid.kernel.torch_level import sigmoid_torch
        function = sigmoid_torch
    elif args.impl == "triton":
        from ops.sigmoid.kernel.triton_level import sigmoid_triton
        function = sigmoid_triton
    elif ":" in args.impl:
        path, entry = args.impl.split(":", 1)
        function = _load(path, entry)
    else:
        raise ValueError(f"unknown impl: {args.impl}")
    device = _resolve_device(args.device)
    golden_spec = yaml.safe_load(
        (HERE / "goldendata/inputs_spec.yaml").read_text(encoding="utf-8"))
    tolerance = golden_spec["tolerance"]
    index = json.loads((Path(args.golden) / "index.json").read_text(
        encoding="utf-8"))

    passed = failed = 0
    worst_abs = worst_rel = 0.0
    for item in index["files"]:
        data = torch.load(Path(args.golden) / item["file"], weights_only=True)
        inputs = [tensor.to(device) for tensor in data["inputs"]]
        out = function(*inputs).float().cpu()
        expected = data["expected"].float()
        diff = (out - expected).abs()
        abs_err = diff.max().item() if diff.numel() else 0.0
        rel_err = (diff / expected.abs().clamp_min(1e-6)).max().item()
        limit = tolerance["pointwise_abs"][data["dtype"]]
        ok = abs_err <= limit
        passed += int(ok)
        failed += int(not ok)
        worst_abs = max(worst_abs, abs_err)
        worst_rel = max(worst_rel, rel_err)
        print(f"{data['case']:12s} {data['dtype']:10s} "
              f"abs={abs_err:.3e} rel={rel_err:.3e} "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"result: {passed} pass / {failed} fail; "
          f"worst abs={worst_abs:.3e}, rel={worst_rel:.3e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
