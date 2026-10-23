#!/usr/bin/env python3
"""A1 aten::matmul interception and native-gradient comparison."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _error(actual, expected):
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    diff = (actual_cpu - expected_cpu).abs()
    return diff.max().item() / max(expected_cpu.abs().max().item(), 1e-6)


def run(profile, backend="triton", with_flaggems=False):
    import torch

    from kernel.triton_level import matmul_triton
    from kernel.torch_level import matmul_torch
    from register import dispatch_key_for_profile, register_a1

    device = profile.torch_device
    cases = [((63, 127), (127, 65)),
             ((2, 33, 65), (2, 65, 47)),
             ((2, 1, 33, 65), (1, 3, 65, 47)),
             ((67,), (67,))]
    inputs = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for i, (left_shape, right_shape) in enumerate(cases):
            generator = torch.Generator().manual_seed(20260929 + i)
            a = torch.randn(left_shape, generator=generator).to(dtype).to(device)
            b = torch.randn(right_shape, generator=generator).to(dtype).to(device)
            native_a = a.detach().requires_grad_(True)
            native_b = b.detach().requires_grad_(True)
            native_out = torch.matmul(native_a, native_b)
            native_out.float().sum().backward()
            inputs.append((a, b, native_out.detach(),
                           native_a.grad.detach(), native_b.grad.detach()))

    if with_flaggems:
        import flag_gems
        flag_gems.enable(unused=["copy_"])

    counter = {}
    key = dispatch_key_for_profile(profile)
    lib = register_a1(key, counter=counter, backend=backend)
    direct_fn = matmul_triton if backend == "triton" else matmul_torch
    worst = 0.0
    for index, (a, b, native, native_grad_a, native_grad_b) in enumerate(inputs):
        a = a.detach().requires_grad_(True)
        b = b.detach().requires_grad_(True)
        before = counter.get("forward", 0)
        hooked = torch.matmul(a, b)
        assert counter.get("forward", 0) > before, (
            f"torch.matmul did not hit aten::matmul on {profile.name} "
            f"with dispatch key {key}"
        )
        direct = direct_fn(a.detach(), b.detach())
        assert torch.equal(hooked.detach(), direct), "A1 path differs from direct kernel"
        hooked.float().sum().backward()
        assert a.grad is not None and b.grad is not None
        errors = (_error(hooked, native), _error(a.grad, native_grad_a),
                  _error(b.grad, native_grad_b))
        if not all(error <= 5e-2 for error in errors):
            from reference import matmul_backward_reference
            cpu_grad_a, cpu_grad_b = matmul_backward_reference(
                a, b, torch.ones_like(native)
            )
            raise AssertionError((
                f"case={index} dtype={a.dtype} shapes={tuple(a.shape)}@{tuple(b.shape)}",
                "vs_native", errors,
                "a1_vs_cpu", (_error(a.grad, cpu_grad_a),
                              _error(b.grad, cpu_grad_b)),
                "native_vs_cpu", (_error(native_grad_a, cpu_grad_a),
                                 _error(native_grad_b, cpu_grad_b)),
            ))
        worst = max(worst, *errors)

    a, b, native = inputs[0][:3]
    before = counter["forward"]
    direct_aten = torch.ops.aten.matmul.default(a, b)
    assert counter["forward"] == before + 1, "direct aten::matmul missed A1"
    assert _error(direct_aten, native) <= 5e-2

    before = counter["forward"]
    cpu_out = torch.matmul(torch.ones(2, 3), torch.ones(3, 2))
    assert torch.equal(cpu_out, torch.full((2, 2), 3.0))
    assert counter["forward"] == before, "CPU matmul was intercepted"

    a, b = inputs[0][:2]
    for grad_left, grad_right in ((True, False), (False, True)):
        left = a.detach().requires_grad_(grad_left)
        right = b.detach().requires_grad_(grad_right)
        before = counter.get("forward", 0)
        torch.matmul(left, right).float().sum().backward()
        assert counter.get("forward", 0) > before
        assert (left.grad is not None) == grad_left
        assert (right.grad is not None) == grad_right

    assert lib is not None
    print(f"A1 matmul ({backend}, {key}, flaggems={with_flaggems}): "
          f"{counter['forward']} interceptions; "
          f"worst native-relative error={worst:.3e}")
    return {"ok": True, "intercepted": counter["forward"],
            "max_native_rel_err": worst}


if __name__ == "__main__":
    from common.device import load_profile
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", nargs="?", default="p800-kunlunxin")
    ap.add_argument("--backend", choices=("triton", "torch"), default="triton")
    ap.add_argument("--with-flaggems", action="store_true")
    args = ap.parse_args()
    print(run(load_profile(args.profile), backend=args.backend,
              with_flaggems=args.with_flaggems))
