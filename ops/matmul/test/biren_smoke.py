#!/usr/bin/env python3
"""Minimal Biren matmul compatibility check for torch-level A1."""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _error(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float()
    return ((actual - expected).abs().max().item()
            / max(expected.abs().max().item(), 1e-6))


def run(profile):
    import torch
    import torch_br  # noqa: F401 - registers the SUPA device
    import br_generator  # noqa: F401 - initializes the Biren JIT runtime

    from kernel.torch_level import matmul_torch
    from reference import matmul_backward_reference, matmul_reference
    from register import dispatch_key_for_profile, register_a1

    device = profile.torch_device
    cases = []
    shapes = [
        ("matrix_tail", (63, 127), (127, 65)),
        ("dot", (67,), (67,)),
        ("matrix_vector", (37, 65), (65,)),
        ("vector_matrix", (65,), (65, 29)),
        ("batch_tail", (2, 33, 65), (2, 65, 47)),
        ("batch_broadcast", (2, 1, 33, 65), (1, 3, 65, 47)),
        ("rank_one", (37, 1), (1, 65)),
    ]
    # The current Biren BR10x SUPA stack rejects FP16 matmul. Keep the
    # supported formats explicit rather than masking the hardware boundary.
    for dtype in (torch.float32, torch.bfloat16):
        for case_index, (name, a_shape, b_shape) in enumerate(shapes):
            generator = torch.Generator().manual_seed(20261124 + case_index)
            a = torch.randn(a_shape, generator=generator, dtype=dtype)
            b = torch.randn(b_shape, generator=generator, dtype=dtype)
            output_shape = torch.matmul(a.float(), b.float()).shape
            grad = torch.randn(output_shape, generator=generator, dtype=dtype)
            cases.append((dtype, name, a, b, grad))

    # Exercise the vendor's ordinary matmul path on the representative 2-D
    # case before installing A1.
    native_cases = [case for case in cases if case[1] == "matrix_tail"]
    for dtype, _, cpu_a, cpu_b, _ in native_cases:
        a, b = cpu_a.to(device), cpu_b.to(device)
        native = torch.matmul(a, b)
        assert _error(native, matmul_reference(cpu_a, cpu_b)) <= 5e-2
    print(f"biren native: {len(native_cases)}/{len(native_cases)} PASS")

    # Direct torch-level implementation before A1 registration.
    for dtype, _, cpu_a, cpu_b, _ in cases:
        a, b = cpu_a.to(device), cpu_b.to(device)
        ref = matmul_reference(cpu_a, cpu_b)
        direct = matmul_torch(a, b)
        assert _error(direct, ref) <= 5e-2, (dtype, "direct")
    print(f"biren direct: {len(cases)}/{len(cases)} PASS "
          "(FP16 unsupported by BR10x)")

    counter = {}
    libs = register_a1(
        dispatch_key_for_profile(profile),
        counter=counter,
        backend="torch",
    )
    mode_hits = {"grad": 0, "no_grad": 0, "inference_mode": 0}
    worst = 0.0
    for dtype, name, cpu_a, cpu_b, cpu_grad in cases:
        a, b = cpu_a.to(device), cpu_b.to(device)
        grad = cpu_grad.to(device)
        ref = matmul_reference(cpu_a, cpu_b)
        ref_a, ref_b = matmul_backward_reference(cpu_a, cpu_b, cpu_grad)

        before = counter.get("forward", 0)
        left = a.detach().requires_grad_(True)
        right = b.detach().requires_grad_(True)
        out = torch.matmul(left, right)
        grad_a, grad_b = torch.autograd.grad(out, (left, right), grad)
        mode_hits["grad"] += counter["forward"] - before
        errors = (_error(out, ref), _error(grad_a, ref_a),
                  _error(grad_b, ref_b))
        assert all(error <= 5e-2 for error in errors), (
            dtype, name, "grad", errors
        )

        before = counter["forward"]
        with torch.no_grad():
            out = torch.matmul(a, b)
        mode_hits["no_grad"] += counter["forward"] - before
        assert _error(out, ref) <= 5e-2, (dtype, name, "no_grad")

        before = counter["forward"]
        with torch.inference_mode():
            out = torch.matmul(a, b)
        mode_hits["inference_mode"] += counter["forward"] - before
        assert _error(out, ref) <= 5e-2, (
            dtype, name, "inference_mode"
        )
        worst = max(worst, *errors)

    before = counter["forward"]
    torch.matmul(torch.ones(2, 3), torch.ones(3, 2))
    assert counter["forward"] == before, "CPU matmul was intercepted"
    expected = len(cases)
    assert mode_hits == {
        "grad": expected,
        "no_grad": expected,
        "inference_mode": expected,
    }, mode_hits
    assert libs is not None
    print(f"biren A1: mode_hits={mode_hits} worst={worst:.3e} "
          "CPU isolation=PASS")
    return {"ok": True, "mode_hits": mode_hits,
            "max_native_rel_err": worst}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile("biren")))
