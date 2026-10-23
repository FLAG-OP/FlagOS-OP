#!/usr/bin/env python3
"""Small application consumer: two matmuls in a trainable forward path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _error(actual, expected):
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return diff.max().item() / max(expected.detach().float().abs().max().item(), 1e-6)


def run(profile, backend="triton"):
    import torch

    from register import dispatch_key_for_profile, register_a1

    class MatmulConsumer(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.activation = torch.nn.GELU()
            self.weight = torch.nn.Parameter(
                torch.randn(11, 5, device=device, dtype=torch.float32) * 0.1
            )

        def forward(self, left, right):
            hidden = torch.matmul(left, right)
            return torch.matmul(self.activation(hidden), self.weight)

    import flag_gems
    flag_gems.only_enable(include=["gelu"])
    assert flag_gems.current_work_registrar is not None
    assert "gelu" in flag_gems.current_work_registrar.include_ops

    device = profile.torch_device
    torch.manual_seed(20260929)
    model = MatmulConsumer(device)
    a = torch.randn(2, 7, 13, device=device).requires_grad_(True)
    b = torch.randn(2, 13, 11, device=device).requires_grad_(True)

    baseline = model(a, b)
    baseline.float().square().mean().backward()
    baseline_grads = (a.grad.detach().clone(), b.grad.detach().clone(),
                      model.weight.grad.detach().clone())
    a.grad = b.grad = model.weight.grad = None

    counter = {}
    key = dispatch_key_for_profile(profile)
    lib = register_a1(key, counter=counter, backend=backend)
    hooked = model(a, b)
    assert counter.get("forward", 0) >= 2, (
        "application consumer did not execute both matmul calls", counter
    )
    hooked.float().square().mean().backward()
    errors = [_error(hooked, baseline)]
    errors.extend(_error(actual, expected) for actual, expected in zip(
        (a.grad, b.grad, model.weight.grad), baseline_grads
    ))
    assert all(error <= 5e-2 for error in errors), errors
    assert lib is not None
    print(f"FlagGems GELU + application consumer ({backend}, {key}): "
          f"calls={counter['forward']} "
          f"output/gradient errors={[round(error, 6) for error in errors]}")
    return {"ok": True, "consumer": "MatmulConsumer", "calls": counter["forward"],
            "max_native_rel_err": max(errors)}


if __name__ == "__main__":
    from common.device import load_profile
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", nargs="?", default="p800-kunlunxin")
    ap.add_argument("--backend", choices=("triton", "torch"), default="triton")
    args = ap.parse_args()
    print(run(load_profile(args.profile), backend=args.backend))
