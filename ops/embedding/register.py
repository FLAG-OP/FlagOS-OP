"""A1 registration for ``aten::embedding`` on XMLIR/CUDA and MLU tensors.

The kernel functions come from the platform facade (``kernel/platform.py``),
which routes each call by tensor device type, so this module serves both
platforms unchanged: Kunlunxin P800 (``AutogradCUDA``) and Cambricon MLU590
(``AutogradPrivateUse1``).
"""
from __future__ import annotations

import torch

# wt 2026-09-23-fix 第二平台: 按 profile/dispatch_key 选择 backend
# （sdpa 的 kernel/backends 选择器同款惯例, 保持单文件轻量形态）
# # wt <wangt635@ustc.edu.cn>
try:
    from .kernel import p800_kunlunxin as _p800
    from .kernel import ascend910 as _ascend
    from .kernel import cambricon as _mlu
except ImportError:
    from kernel import p800_kunlunxin as _p800
    from kernel import ascend910 as _ascend
    from kernel import cambricon as _mlu

_BACKENDS = {
    "p800-kunlunxin": _p800,
    "ascend910": _ascend,
    "cambricon": _mlu,
}


def _backend_for(dispatch_key: str, platform=None):
    if platform and platform in _BACKENDS:
        return _BACKENDS[platform]
    by_key = {"AutogradCUDA": _p800, "CUDA": _p800,
              "AutogradPrivateUse1": _ascend, "PrivateUse1": _ascend,
              # wt 2026-10-03-fix cambricon: 与 configs/devices/cambricon.yaml
              # 的 PrivateUse1 不同——A1 带 autograd 交付须注册 Autograd key
              # （见 reports/cambricon.md §4）; ascend 与 mlu 同 key 时
              # profile/platform 显式指定优先
              "Autograd": _mlu}
    if dispatch_key not in by_key:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: "
            f"{list(by_key)}（或显式传 platform）")
    return by_key[dispatch_key]


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


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None):
    """Override Python ``F.embedding``/``torch.embedding`` dispatch."""
    backend = _backend_for(dispatch_key, platform)
    # _EmbeddingA1Function 的 staticmethod 读模块全局, 三者均须 global
    global PLATFORM, embedding, embedding_backward
    PLATFORM = backend.PLATFORM
    embedding = backend.embedding
    embedding_backward = backend.embedding_backward
    # 注册守卫: 跨平台误用在注册时拦截，而不是运行时静默错
    # （与 ops/sdpa/register.py 同口径; 能力探测按 backend 声明）。
    # wt 2026-10-03-fix 能力探测: backend 模块自带 _DEVICE_PROBE
    # (attr, fn) 声明; 无声明则跳过探测（向后兼容旧 backend）
    # # wt <wangt635@ustc.edu.cn>
    probe = getattr(backend, "_DEVICE_PROBE", None)
    if probe is not None:
        attr, fn = probe
        if not (hasattr(torch, attr) and getattr(torch, attr) is not None
                and fn()):
            raise RuntimeError(
                f"register_a1 binding={PLATFORM!r}, "
                f"dispatch_key={dispatch_key!r} requires an available "
                f"{backend.PLATFORM} device")

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
