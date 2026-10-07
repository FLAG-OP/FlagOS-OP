#!/usr/bin/env python3
"""A1 op-level test: interception, forward equality, and autograd."""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

_CALLS = {"n": 0}


def _make(dev, dtype, requires_grad=False):
    import torch

    torch.manual_seed(7)
    weight = (torch.randn(32, 16) * 0.1).to(dtype).to(dev)
    if requires_grad:
        weight = weight.requires_grad_(True)
    indices = torch.tensor([[0, 1, 1, 3, 3, 31]], device=dev)
    return weight, indices


def run(profile):
    import torch
    import torch.nn.functional as F

    from kernel.platform import embedding as direct_embedding
    from reference import embedding_backward_reference
    from register import register_a1

    dev = profile.torch_device
    dtype = torch.float16
    F = torch.nn.functional

    # Native baseline must be captured before process-wide A1 registration.
    weight, indices = _make(dev, dtype, requires_grad=True)
    baseline = F.embedding(indices, weight, padding_idx=3)
    baseline.float().sum().backward()
    baseline_grad = weight.grad.detach().clone()

    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)

    weight, indices = _make(dev, dtype)
    hooked = F.embedding(indices, weight, padding_idx=3)
    direct = direct_embedding(weight, indices, padding_idx=3)
    assert _CALLS["n"] > 0, "A1 embedding interception failed"
    assert torch.equal(hooked, direct), "hooked path differs from direct path"

    # Registered autograd path versus native dense backward.
    weight_g, indices_g = _make(dev, dtype, requires_grad=True)
    out = F.embedding(
        indices_g, weight_g, padding_idx=3, scale_grad_by_freq=False
    )
    assert out.requires_grad
    out.float().sum().backward()
    assert weight_g.grad is not None
    err = (weight_g.grad.float() - baseline_grad.float()).abs().max().item()
    assert err < 2e-3, err

    # Native backward semantics must match CPU on both platforms.  P800 XPU
    # native backward lacks scale_grad_by_freq, so the A1 wrapper supplies
    # inverse-frequency scaling there; Cambricon native implements it
    # directly.  Both are compared against the CPU reference below.
    weight_s, indices_s = _make(dev, dtype, requires_grad=True)
    grad = (torch.randn(*indices_s.shape, weight_s.shape[-1]) * 0.1)
    grad = grad.to(dtype).to(dev)
    out_s = F.embedding(
        indices_s, weight_s, padding_idx=3, scale_grad_by_freq=True
    )
    out_s.backward(grad)
    ref_s = embedding_backward_reference(
        grad.cpu(), indices_s.cpu(), weight_s.shape[0], 3, True, False
    )
    scale_err = (
        weight_s.grad.float().cpu() - ref_s.float()
    ).abs().max().item()
    assert scale_err < 2e-3, scale_err

    # Forward accepts sparse=True because the lookup itself is unchanged.
    # The unsupported boundary is sparse backward / sparse gradient format.
    sparse_forward = F.embedding(indices, weight, sparse=True)
    assert torch.equal(sparse_forward, direct)
    weight_sparse, indices_sparse = _make(dev, dtype, requires_grad=True)
    sparse_out = F.embedding(indices_sparse, weight_sparse, sparse=True)
    try:
        sparse_out.float().sum().backward()
    except NotImplementedError:
        sparse_backward_guard = True
    else:
        sparse_backward_guard = False
    assert sparse_backward_guard

    print(
        f"  interception count={_CALLS['n']}; hooked=direct bitwise; "
        f"dense grad err={err:.3e}; scale_freq grad err={scale_err:.3e}; "
        f"sparse fwd OK; sparse bwd rejected"
    )
    return {
        "ok": True,
        "intercepted": _CALLS["n"],
        "bitwise_vs_direct": True,
        "dense_grad_err": round(err, 8),
        "scale_grad_err": round(scale_err, 8),
        "sparse_forward": True,
        "sparse_backward_guard": True,
    }


if __name__ == "__main__":
    from _profile import load_profile

    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
