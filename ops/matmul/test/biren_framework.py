#!/usr/bin/env python3
"""Small Biren application consumer without FlagGems dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _error(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    return ((actual - expected).abs().max().item()
            / max(expected.abs().max().item(), 1e-6))


def run(profile):
    import torch
    import torch_br  # noqa: F401
    import br_generator  # noqa: F401

    from register import dispatch_key_for_profile, register_a1

    class Consumer(torch.nn.Module):
        def __init__(self, device="cpu"):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(11, 5, dtype=torch.float32).to(device) * 0.1
            )

        def forward(self, left, right):
            hidden = torch.matmul(left, right)
            return torch.matmul(torch.nn.functional.gelu(hidden), self.weight)

    torch.manual_seed(20261124)
    reference_model = Consumer("cpu")
    reference_a = torch.randn(2, 7, 13).requires_grad_(True)
    reference_b = torch.randn(2, 13, 11).requires_grad_(True)
    baseline = reference_model(reference_a, reference_b)
    baseline.float().square().mean().backward()
    expected_grads = (
        reference_a.grad.detach().clone(),
        reference_b.grad.detach().clone(),
        reference_model.weight.grad.detach().clone(),
    )

    device = profile.torch_device
    model = Consumer(device)
    model.load_state_dict(reference_model.state_dict())
    a = reference_a.detach().to(device).requires_grad_(True)
    b = reference_b.detach().to(device).requires_grad_(True)
    counter = {}
    libs = register_a1(
        dispatch_key_for_profile(profile),
        counter=counter,
        backend="torch",
    )
    a.grad = b.grad = model.weight.grad = None
    hooked = model(a, b)
    hooked.float().square().mean().backward()
    errors = [_error(hooked, baseline)]
    errors.extend(_error(got, expected) for got, expected in zip(
        (a.grad, b.grad, model.weight.grad),
        expected_grads,
    ))
    assert counter["forward"] >= 2, counter
    assert max(errors) <= 5e-2, errors
    assert libs is not None
    print(f"biren framework: calls={counter['forward']} "
          f"max_error={max(errors):.3e}")
    return {"ok": True, "calls": counter["forward"],
            "max_native_rel_err": max(errors)}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile("biren")))
