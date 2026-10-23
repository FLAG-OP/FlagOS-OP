#!/usr/bin/env python3
"""Check matmul forward and both gradients against verified CPU goldens."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))


def _device(value):
    if value == "cpu" or ":" in value:
        return value
    from common.device import load_profile
    return load_profile(value).torch_device


def _relative_error(actual, expected):
    diff = (actual.detach().float().cpu() - expected.float()).abs()
    if diff.numel() == 0:
        return 0.0, False
    scale = expected.float().abs().max().item()
    if scale == 0:
        return diff.max().item(), True
    return diff.max().item() / scale, False


def _run(impl, a, b, grad_output):
    if impl == "triton":
        from kernel.triton_level import matmul_backward_triton, matmul_triton
        out = matmul_triton(a, b)
        grad_a, grad_b = matmul_backward_triton(a, b, grad_output)
    elif impl == "torch":
        from kernel.torch_level import matmul_torch
        left = a.detach().requires_grad_(True)
        right = b.detach().requires_grad_(True)
        out = matmul_torch(left, right)
        grad_a, grad_b = torch.autograd.grad(out, (left, right), grad_output)
    elif impl in ("native", "a1-torch", "a1-triton"):
        left = a.detach().requires_grad_(True)
        right = b.detach().requires_grad_(True)
        out = torch.matmul(left, right)
        grad_a, grad_b = torch.autograd.grad(out, (left, right), grad_output)
    else:
        raise ValueError(f"unknown implementation: {impl}")
    return out, grad_a, grad_b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", choices=("triton", "torch", "native",
                                       "a1-torch", "a1-triton"), required=True,
                    help="torch is a composition diagnostic; triton/a1-triton test the custom kernel")
    ap.add_argument("--device", required=True, help="profile name or torch device")
    ap.add_argument("--golden", type=Path, default=HERE / "goldendata")
    ap.add_argument("--exclude-case", action="append", default=[],
                    help="explicitly exclude a named golden case")
    ap.add_argument("--include-case", action="append", default=[],
                    help="run only named golden cases (repeatable)")
    args = ap.parse_args()
    device = _device(args.device)
    counter = None
    if args.impl in ("a1-torch", "a1-triton"):
        if args.device == "cpu" or ":" in args.device:
            raise ValueError(f"{args.impl} requires a named device profile")
        from common.device import load_profile
        from register import dispatch_key_for_profile, register_a1
        profile = load_profile(args.device)
        counter = {}
        lib = register_a1(dispatch_key_for_profile(profile), counter=counter,
                          backend="triton" if args.impl == "a1-triton" else "torch")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    known = {case["name"] for case in spec["cases"]}
    unknown = (set(args.exclude_case) | set(args.include_case)) - known
    if unknown:
        raise ValueError(f"unknown golden cases: {sorted(unknown)}")
    tolerance = float(spec["tolerance"]["gemm_rel"])
    index = json.loads((args.golden / "index.json").read_text())
    if index["op"] != "matmul":
        raise ValueError("golden index is for a different operator")
    names = [item["file"] for item in index["files"]]
    if len(names) != len(set(names)):
        raise ValueError("golden index contains duplicate filenames")

    passed = failed = 0
    worst = 0.0
    for item in index["files"]:
        path = args.golden / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"golden checksum mismatch: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (args.include_case and data["case"] not in args.include_case) or (
                data["case"] in args.exclude_case):
            continue
        a, b = (tensor.to(device) for tensor in data["inputs"])
        grad_output = data["grad_output"].to(device)
        before = counter.get("forward", 0) if counter is not None else None
        out, grad_a, grad_b = _run(args.impl, a, b, grad_output)
        if counter is not None and counter.get("forward", 0) != before + 1:
            raise AssertionError(f"A1 interception missing for {item['file']}")
        checks = [
            _relative_error(out, data["expected"]),
            _relative_error(grad_a, data["grad_a"]),
            _relative_error(grad_b, data["grad_b"]),
        ]
        errors = [metric for metric, _ in checks]
        worst = max(worst, *errors)
        zero_tol = float(spec["tolerance"]["zero_abs"][data["dtype"]])
        ok = all(metric <= (zero_tol if zero else tolerance)
                 for metric, zero in checks)
        passed += int(ok)
        failed += int(not ok)
        print(f"{item['file']:<48} fwd={errors[0]:.2e} "
              f"dA={errors[1]:.2e} dB={errors[2]:.2e} "
              f"{'PASS' if ok else 'FAIL'}")

    if passed + failed == 0:
        raise ValueError("no golden cases selected")
    calls = f"; A1 calls={counter['forward']}" if counter is not None else ""
    if args.exclude_case:
        print(f"excluded cases: {', '.join(args.exclude_case)}")
    print(f"{args.impl} @ {device}: {passed} pass / {failed} fail; "
          f"worst={worst:.2e}{calls}")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
