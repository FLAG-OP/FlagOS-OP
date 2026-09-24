#!/usr/bin/env python3
"""Kernel-level embedding validation: forward, backward, sentinel, perf."""
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _forward_cases():
    return [
        # (num_weights, dim, index shape, padding_idx, tag)
        (32, 8, (7,), -1, "1d"),
        (32, 8, (3, 17), 3, "2d-padding"),
        (32, 8, (2, 3, 5), -1, "3d"),
        (128, 80, (33,), -1, "d80-tail"),
        (128, 512, (17,), 127, "d512"),
        (16, 8, (0,), -1, "empty"),
    ]


def _backward_case(device, dtype, scale):
    import torch

    torch.manual_seed(42)
    num_weights, dim = 16, 12
    indices = torch.tensor([[0, 1, 1, 3, 3, 3]], device=device)
    weight = (torch.randn(num_weights, dim) * 0.1).to(dtype).to(device)
    grad = (torch.randn(*indices.shape, dim) * 0.1).to(dtype).to(device)
    return weight, indices, grad, num_weights, 3, scale


def run(profile):
    import torch

    from kernel.p800_kunlunxin import embedding, embedding_backward
    from reference import embedding_backward_reference, embedding_reference

    dev = profile.torch_device
    results = []
    max_forward_err = 0.0

    for num_weights, dim, index_shape, padding_idx, tag in _forward_cases():
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            torch.manual_seed(1234)
            weight = (torch.randn(num_weights, dim) * 0.1).to(dtype).to(dev)
            indices = torch.randint(
                0, num_weights, index_shape, device=dev
            )
            out = embedding(weight, indices, padding_idx, False, False)
            ref = embedding_reference(
                weight.cpu(), indices.cpu(), padding_idx
            )
            diff = (out.float().cpu() - ref.float()).abs()
            err = diff.max().item() if diff.numel() else 0.0
            max_forward_err = max(max_forward_err, err)
            assert err == 0.0, (tag, dtype, err)
            results.append(
                f"[OK] fwd {tag:12s} {str(dtype).split('.')[-1]:8s} "
                f"shape={tuple(out.shape)}"
            )

    # int32 is accepted by the current XMLIR native row gather.
    weight = torch.randn(32, 8, device=dev, dtype=torch.float16)
    indices = torch.tensor([1, 4, 8], device=dev, dtype=torch.int32)
    assert torch.equal(
        embedding(weight, indices).cpu(),
        embedding_reference(weight.cpu(), indices.cpu()),
    )
    results.append("[OK] fwd indices=int32")

    max_backward_err = 0.0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for scale in (False, True):
            weight, indices, grad, num_weights, pad, _ = _backward_case(
                dev, dtype, scale
            )
            out = embedding_backward(
                grad, indices, num_weights, pad, scale, False
            )
            ref = embedding_backward_reference(
                grad.cpu(), indices.cpu(), num_weights, pad, scale, False
            )
            err = (out.float().cpu() - ref.float()).abs().max().item()
            tol = 1e-6 if dtype == torch.float32 else 2e-3
            max_backward_err = max(max_backward_err, err)
            assert err < tol, (dtype, scale, err)
            results.append(
                f"[OK] bwd {str(dtype).split('.')[-1]:8s} "
                f"scale_freq={int(scale)} err={err:.3e}"
            )

    # Sentinel and boundary checks.
    weight = torch.randn(32, 8, device=dev, dtype=torch.float16)
    indices = torch.tensor([[1, 4, 4], [8, 8, 31]], device=dev)
    out1 = embedding(weight, indices)
    out2 = embedding(weight, indices)
    assert torch.equal(out1, out2)
    assert not torch.equal(out1, embedding(weight + 0.125, indices))
    results.append("[OK] sentinel: deterministic + input-sensitive")

    # ``sparse=True`` requests a sparse gradient representation but does not
    # change the forward lookup.  The forward is therefore accepted; backward
    # is the explicit unsupported boundary.
    sparse_forward = embedding(weight, indices, sparse=True)
    assert torch.equal(sparse_forward, embedding(weight, indices))
    results.append("[OK] sparse=true forward accepted as dense lookup")

    grad = torch.randn_like(weight)
    try:
        from kernel.p800_kunlunxin import embedding_backward
        embedding_backward(grad, indices, weight.shape[0], -1, False, True)
    except NotImplementedError:
        results.append("[OK] guard: sparse backward rejected")
    else:
        raise AssertionError("sparse backward should be rejected")

    try:
        embedding(weight, indices, padding_idx=32)
    except IndexError:
        results.append("[OK] guard: invalid padding_idx rejected")
    else:
        raise AssertionError("invalid padding_idx should be rejected")

    perf = _quick_perf(dev)
    for line in results:
        print(line)
    return {
        "ok": True,
        "forward_cases": 20,
        "backward_cases": 6,
        "max_forward_err": max_forward_err,
        "max_backward_err": round(max_backward_err, 8),
        "perf_ms_16k_d128": perf,
    }


def _consume(output):
    return output.reshape(-1)[0].item()


def _quick_perf(dev):
    import torch

    from kernel.p800_kunlunxin import embedding

    torch.manual_seed(7)
    weight = torch.randn(4096, 128, device=dev, dtype=torch.float16) * 0.1
    indices = torch.randint(0, 4096, (16384,), device=dev)
    for _ in range(5):
        _consume(embedding(weight, indices))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _consume(embedding(weight, indices))
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / 20 * 1000, 4)


if __name__ == "__main__":
    from _profile import load_profile

    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
