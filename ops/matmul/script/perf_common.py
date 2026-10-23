"""Deterministic matmul performance inputs and forward/backward callables."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from ops.matmul.kernel._shape import matrix_inputs
from ops.matmul.reference import matmul_backward_reference, matmul_reference
from ops.matmul.script.gen_golden import make_case

SPEC_PATH = Path(__file__).resolve().parents[1] / "goldendata/inputs_spec.yaml"
# Empty/zero-K shapes remain correctness tests; no GEMM is executed for them.
NO_GEMM = {"zero_k", "empty_m", "empty_n", "empty_batch"}


def workloads():
    """One fixed seed (the first golden seed) per nonempty shape and dtype."""
    spec = yaml.safe_load(SPEC_PATH.read_text())
    for ci, case in enumerate(spec["cases"]):
        if case["name"] in NO_GEMM:
            continue
        for di, dtype in enumerate(spec["dtypes"]):
            seed = spec["seed_base"] + (ci * len(spec["dtypes"]) + di) * spec["seeds_per_case"]
            yield case, dtype, seed


def _error(actual, expected):
    got = actual.detach().float().cpu()
    ref = expected.float()
    if got.shape != ref.shape or not torch.isfinite(got).all().item():
        raise AssertionError("invalid shape or nonfinite performance result")
    scale = ref.abs().max().item() if ref.numel() else 0.0
    err = (got - ref).abs().max().item() if ref.numel() else 0.0
    return (err / scale if scale else err), scale == 0


def prepare(case, dtype_name, seed, impl, direction, device):
    """Prepare/check outside timing; backward measures dA+dB without forward."""
    from ops.matmul.kernel.triton_level import matmul_backward_triton, matmul_triton

    dtype = getattr(torch, dtype_name)
    ca, cb, cg = make_case(case, dtype, seed)
    a, b, grad = (t.to(device) for t in (ca, cb, cg))
    left, right, batch, _, _ = matrix_inputs(ca, cb)
    flops = 2 * math.prod(batch) * left.shape[-2] * left.shape[-1] * right.shape[-1]
    if direction == "forward":
        expected = (matmul_reference(ca, cb),)
        if impl == "triton":
            compute = lambda: (matmul_triton(a, b),)
        elif impl == "native":
            compute = lambda: (torch.matmul(a, b),)
        elif impl == "flaggems":
            from flag_gems import ops
            if a.ndim == b.ndim == 3 and a.shape[0] == b.shape[0]:
                fn = ops.bmm
            elif a.ndim == b.ndim == 2:
                fn = ops.mm
            else:
                raise ValueError("no full matmul entry in this FlagGems version")
            compute = lambda: (fn(a, b),)
        else:
            raise ValueError(impl)
    elif direction == "backward":
        expected = matmul_backward_reference(ca, cb, cg)
        flops *= 2  # Two logical GEMMs; excludes broadcast-reduction FLOPs.
        if impl == "triton":
            compute = lambda: matmul_backward_triton(a, b, grad)
        elif impl == "native":
            a.requires_grad_(True)
            b.requires_grad_(True)
            output = torch.matmul(a, b)
            # Reuse the forward graph; autograd.grad does not accumulate .grad.
            compute = lambda: torch.autograd.grad(output, (a, b), grad, retain_graph=True)
        else:
            raise ValueError("FlagGems backward is not a validated comparison path")
    else:
        raise ValueError(direction)

    errors = [_error(got, ref) for got, ref in zip(compute(), expected)]
    zero_tol = 1e-5 if dtype == torch.float32 else 1e-2
    if not all(e <= (zero_tol if zero else 5e-2) for e, zero in errors):
        raise AssertionError((case["name"], dtype_name, impl, direction, errors))

    def call():
        results = compute()
        # Read both gradients to ensure every timed backward call completes.
        for result in results:
            if result.numel():
                result.reshape(-1)[0].item()

    return call, {"flops": flops, "max_check_error": max(e for e, _ in errors)}


def perf_cases(profile):
    from common.perf import PerfCase

    result = []
    for case, dtype, seed in workloads():
        for direction in ("forward", "backward"):
            for impl in ("triton", "native"):
                state = {}

                def make(p, case=case, dtype=dtype, seed=seed,
                         impl=impl, direction=direction, state=state):
                    fn, info = prepare(case, dtype, seed, impl, direction, p.torch_device)
                    state.update(info)
                    return fn

                def derived(ms, state=state):
                    return {"GFLOPS": state["flops"] / ms / 1e6,
                            "max_check_error": state["max_check_error"]}

                result.append(PerfCase(
                    f"ops.matmul.{case['name']}.{dtype}.{direction}.{impl}",
                    group="ops", level="kernel", make_fn=make, derived=derived))
    return result
