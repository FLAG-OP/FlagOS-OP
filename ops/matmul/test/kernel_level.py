#!/usr/bin/env python3
"""Direct matmul forward/backward correctness and sentinel checks."""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _error(actual, expected):
    diff = (actual.detach().float().cpu() - expected.float()).abs()
    if diff.numel() == 0:
        return 0.0
    scale = expected.float().abs().max().item()
    return diff.max().item() / scale if scale else diff.max().item()


def _check_k1_sram_tail(profile):
    """A preceding FP32 dot must not contaminate BF16 masked K lanes."""
    import torch
    import triton
    from flag_gems.runtime import torch_device_fn
    from kernel.triton_level import _matmul_kernel

    # Allocate everything first; launch prime -> target without an intervening
    # tensor operation that might overwrite the SRAM and hide the regression.
    primes = [
        (torch.full((128, 32), value, device=profile.torch_device),
         torch.full((32, 128), value, device=profile.torch_device),
         torch.empty((128, 128), device=profile.torch_device))
        for value in (0.0, 1.0, 3.0)
    ]
    targets = [
        (k, value,
         torch.full((37, k), value, dtype=torch.bfloat16, device=profile.torch_device),
         torch.full((k, 65), value, dtype=torch.bfloat16, device=profile.torch_device),
         torch.empty((37, 65), dtype=torch.bfloat16, device=profile.torch_device))
        for k in (1, 2) for value in (0.0, 1.0)
    ]

    def launch(a, b, out):
        m, k = a.shape
        n = b.shape[1]
        _matmul_kernel[(triton.cdiv(m, 32) * triton.cdiv(n, 32), 1)](
            a, b, out, m, n, k, 0, *a.stride(), 0, *b.stride(),
            0, *out.stride(), BM=32, BN=32, BK=32, num_warps=4)

    count = 0
    with torch_device_fn.device(profile.torch_device):
        for prime in primes:
            launch(*prime)
        for _, _, a, b, out in targets:
            launch(a, b, out)
        targets[-1][-1].cpu()
        for prime in primes:
            for k, value, a, b, out in targets:
                for _ in range(3):
                    launch(*prime)
                    launch(a, b, out)
                    got = out.float().cpu()
                    assert torch.all(got == k * value * value).item(), (
                        "BF16 masked K tail contains preceding SRAM values", k, value,
                        (got - k * value * value).abs().max().item())
                    count += 1
    print(f"kernel: BF16 K=1/2 SRAM contamination regression {count}/{count} PASS")


def run(profile):
    import torch
    import yaml

    from kernel.triton_level import matmul_backward_triton, matmul_triton
    from reference import matmul_backward_reference, matmul_reference
    from script.gen_golden import make_case

    if profile.vendor == "kunlunxin":
        _check_k1_sram_tail(profile)

    spec = yaml.safe_load((OP_DIR / "goldendata/inputs_spec.yaml").read_text())
    rel_tol = float(spec["tolerance"]["gemm_rel"])
    count = 0
    worst_forward = worst_backward = 0.0
    for case in spec["cases"]:
        for dtype_name in spec["dtypes"]:
            dtype = getattr(torch, dtype_name)
            zero_tol = float(spec["tolerance"]["zero_abs"][dtype_name])
            for seed_no in range(spec["seeds_per_case"]):
                seed = spec["seed_base"] + count
                cpu_a, cpu_b, cpu_grad = make_case(case, dtype, seed)
                a, b, grad = (tensor.to(profile.torch_device)
                              for tensor in (cpu_a, cpu_b, cpu_grad))
                original_a, original_b = a.clone(), b.clone()
                expected = matmul_reference(cpu_a, cpu_b)
                ref_a, ref_b = matmul_backward_reference(
                    cpu_a, cpu_b, cpu_grad
                )
                out = matmul_triton(a, b)
                grad_a, grad_b = matmul_backward_triton(a, b, grad)
                assert out.shape == expected.shape
                assert grad_a.shape == a.shape and grad_b.shape == b.shape
                assert torch.equal(a, original_a) and torch.equal(b, original_b), (
                    case["name"], "an input was modified"
                )
                errors = (
                    _error(out, expected), _error(grad_a, ref_a),
                    _error(grad_b, ref_b),
                )
                refs = (expected, ref_a, ref_b)
                for err, ref in zip(errors, refs):
                    ref_scale = (ref.float().abs().max().item()
                                 if ref.numel() else 0.0)
                    threshold = zero_tol if ref_scale == 0 else rel_tol
                    assert err <= threshold, (case["name"], dtype_name, seed, err)
                worst_forward = max(worst_forward, errors[0])
                worst_backward = max(worst_backward, *errors[1:])
                count += 1

    a = torch.randn(2, 33, 65, device=profile.torch_device,
                    dtype=torch.float32)
    b = torch.randn(2, 65, 47, device=profile.torch_device,
                    dtype=torch.float32)
    first = matmul_triton(a, b)
    assert torch.equal(first, matmul_triton(a, b)), "matmul is not deterministic"
    assert not torch.equal(first, matmul_triton(a + 1, b)), "input-insensitive output"

    # sum() backward supplies an expanded gradient with zero matrix strides.
    left = torch.randn(63, 127, device=profile.torch_device)
    right = torch.randn(127, 65, device=profile.torch_device)
    grad = torch.ones(1, 1, device=profile.torch_device).expand(63, 65)
    assert grad.stride() == (0, 0)
    got_a, got_b = matmul_backward_triton(left, right, grad)
    ref_a, ref_b = matmul_backward_reference(left.cpu(), right.cpu(), grad.cpu())
    assert _error(got_a, ref_a) <= rel_tol
    assert _error(got_b, ref_b) <= rel_tol

    for left, right in ((torch.empty(2, device=profile.torch_device),
                         torch.empty(3, device=profile.torch_device)),
                        (torch.empty((), device=profile.torch_device),
                         torch.empty(1, device=profile.torch_device))):
        try:
            matmul_triton(left, right)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid matmul shape was accepted")

    print(f"kernel: {count} forward/backward cases; "
          f"worst_fwd={worst_forward:.3e} worst_bwd={worst_backward:.3e}; "
          "sentinel and zero-stride backward PASS")
    return {"ok": True, "cases": count, "max_forward_err": worst_forward,
            "max_backward_err": worst_backward, "sentinel": True}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
