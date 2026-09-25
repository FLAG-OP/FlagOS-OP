"""A1 registration for ``aten::embedding`` on XMLIR/CUDA and MLU tensors.

The kernel functions come from the platform facade (``kernel/platform.py``),
which routes each call by tensor device type, so this module serves both
platforms unchanged: Kunlunxin P800 (``AutogradCUDA``) and Cambricon MLU590
(``AutogradPrivateUse1``).
"""
from __future__ import annotations

import torch

try:
    from .kernel.platform import PLATFORM
    from .kernel.platform import embedding, embedding_backward
except ImportError:
    from kernel.platform import PLATFORM
    from kernel.platform import embedding, embedding_backward


class _EmbeddingA1Function(torch.autograd.Function):
    """Native row-gather forward plus dense semantic backward."""

    @staticmethod
    def forward(ctx, weight, indices, padding_idx, scale_grad_by_freq,
                sparse):
        ctx.save_for_backward(indices)
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return embedding(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_weight = embedding_backward(
            grad_output, indices, ctx.num_weights, ctx.padding_idx,
            ctx.scale_grad_by_freq, ctx.sparse,
        )
        return grad_weight, None, None, None, None


def register_a1(dispatch_key: str = "AutogradCUDA", counter: dict | None = None):
    """Override Python ``F.embedding``/``torch.embedding`` dispatch."""
    # 注册守卫: 跨平台误用在注册时拦截，而不是运行时静默错
    # （与 ops/sdpa/register.py 同口径）。
    if dispatch_key in ("CUDA", "AutogradCUDA") and not torch.cuda.is_available():
        raise RuntimeError(
            f"register_a1 binding={PLATFORM!r}, dispatch_key={dispatch_key!r} "
            "requires an available CUDA/XMLIR device"
        )
    if dispatch_key in ("PrivateUse1", "AutogradPrivateUse1") and not (
        hasattr(torch, "mlu") and torch.mlu.is_available()
    ):
        raise RuntimeError(
            f"register_a1 binding={PLATFORM!r}, dispatch_key={dispatch_key!r} "
            "requires an available torch_mlu/MLU device"
        )

    def impl(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
             sparse=False):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return _EmbeddingA1Function.apply(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("embedding", impl, dispatch_key)
    return lib
