"""A1 registration for aten::matmul; registration lives for the process."""
from __future__ import annotations

import os
from importlib import import_module

import torch

# tyf-2026-10-23-fix: load hardware dependencies only when registering a backend.
# tyf <tyanfeng@mail.ustc.edu.cn>


def dispatch_key_for_profile(profile) -> str:
    # P800's CUDA key catches forward but leaves aten::matmul_backward broken.
    if profile.vendor == "kunlunxin" and profile.dispatch_key == "CUDA":
        return "AutogradCUDA"
    return profile.dispatch_key


class _MatmulA1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, backend):
        forward, ctx.backward_impl = backend
        ctx.save_for_backward(a, b)
        return forward(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("matmul A1 does not support higher-order gradients")
        a, b = ctx.saved_tensors
        grad_a, grad_b = ctx.backward_impl(
            a, b, grad_output, *ctx.needs_input_grad[:2]
        )
        return grad_a, grad_b, None


def register_a1(dispatch_key: str, counter: dict | None = None,
                backend: str = "triton", hit_log: str | None = None,
                hit_marker: str | None = None):
    if backend not in ("triton", "torch"):
        raise ValueError(f"unsupported matmul backend: {backend}")

    # Select package vs standalone imports explicitly. A missing dependency must
    # retain its original exception, not trigger an unrelated import fallback.
    prefix = f"{__package__}.kernel" if __package__ else "kernel"
    module = import_module(f"{prefix}.{backend}_level")
    implementation = (getattr(module, f"matmul_{backend}"),
                      getattr(module, f"matmul_backward_{backend}"))

    def impl(self, other):
        if counter is not None:
            counter["forward"] = counter.get("forward", 0) + 1
        if hit_log and (hit_marker is None or os.path.exists(hit_marker)):
            with open(hit_log, "a", encoding="utf-8") as log:
                log.write(f"{os.getpid()} {tuple(self.shape)} {tuple(other.shape)}\n")
        return _MatmulA1Function.apply(self, other, implementation)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("matmul", impl, dispatch_key)
    return lib
