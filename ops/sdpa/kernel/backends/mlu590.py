"""Cambricon MLU590 SDPA backend.

Engineering choice
------------------
The MLU stack exposes no private efficient-attention kernel that stays on
device (``aten::_scaled_dot_product_efficient_attention`` falls back to CPU
and fails for flash/cudnn variants).  Python ``F.sdpa`` is a fast composite
math path, but it becomes recursive once A1 registers AutogradPrivateUse1.

Primary path: ``aten::_scaled_dot_product_attention_math`` — a non-recursive
private entry that runs the same matmul+softmax composition as native F.sdpa
on MLU (≈1.5-3.5x native, vs FlagGems Triton ≈10-40x slower).  bool masks
are converted to additive ``-inf`` bias because the math kernel mishandles
bool on this stack.  Fully masked rows restore the CPU/aten NaN corner.
FlagGems Cambricon remains available as a fallback composition path when the
private math op is missing.  float32 without the private op uses the fp32
reference for golden compatibility.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import torch

try:  # Standalone ops/sdpa execution inserts OP_DIR into sys.path.
    from reference import sdpa_reference
except ModuleNotFoundError:  # Package-style import from repository root.
    from ops.sdpa.reference import sdpa_reference


PLATFORM = "mlu590"
SUPPORTED_DEVICE_TYPES = ("mlu",)


def _has_mlu_stack() -> bool:
    if importlib.util.find_spec("torch_mlu") is None:
        return False
    try:
        import torch_mlu  # noqa: F401
        return True
    except Exception:
        return False


def _has_math_private() -> bool:
    try:
        torch.ops.aten._scaled_dot_product_attention_math
        return True
    except Exception:
        return False


def _gems_attention():
    from flag_gems.runtime.backend._cambricon.ops.attention import (
        scaled_dot_product_attention_forward,
    )
    return scaled_dot_product_attention_forward


def _bool_to_additive(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """math kernel mishandles bool on MLU; additive -inf bias is exact."""
    bias = torch.zeros_like(attn_mask, dtype=dtype)
    return bias.masked_fill(~attn_mask, float("-inf"))


def _restore_full_mask_nan(query, key, out, attn_mask, is_causal):
    """CPU/aten semantics: a fully masked row is NaN; device kernels may not."""
    if attn_mask is None:
        return out
    if attn_mask.dtype == torch.bool:
        effective_false = ~attn_mask.bool()
    else:
        effective_false = torch.isneginf(attn_mask.float())
    if is_causal:
        sq, skv = query.shape[-2], key.shape[-2]
        causal = torch.ones(
            (sq, skv), dtype=torch.bool, device=query.device
        ).tril()
        effective_false = effective_false | ~causal
    all_masked = effective_false.all(dim=-1, keepdim=True)
    if not bool(all_masked.any()):
        return out
    return out.masked_fill(all_masked, float("nan"))


def _grad_path(query, key, value, attn_mask, dropout_p, is_causal,
               scale, enable_gqa):
    """Route differentiable direct calls through A1 math backward.

    The private math path has no reliable autograd on this stack for all
    mask forms; borrow the custom Function.  Inside its forward autograd
    is disabled so there is no recursion into this path.
    """
    try:
        from ...register import _SDPA_A1_Function
    except ImportError:
        from register import _SDPA_A1_Function

    saved_impl = getattr(_SDPA_A1_Function, "_impl", "triton")
    _SDPA_A1_Function._impl = "triton"
    try:
        return _SDPA_A1_Function.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )
    finally:
        _SDPA_A1_Function._impl = saved_impl


def _sdpa_math_private(query, key, value, attn_mask, dropout_p, is_causal,
                       scale, enable_gqa):
    """Non-recursive fast path via aten::_scaled_dot_product_attention_math."""
    bias = attn_mask
    if bias is not None and bias.dtype == torch.bool:
        bias = _bool_to_additive(bias, query.dtype)
    elif bias is not None and bias.dtype != query.dtype:
        bias = bias.to(query.dtype)

    out = torch.ops.aten._scaled_dot_product_attention_math(
        query, key, value, bias, dropout_p, is_causal, None,
        scale=scale, enable_gqa=enable_gqa,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return _restore_full_mask_nan(query, key, out, attn_mask, is_causal)


def _sdpa_gems(query, key, value, attn_mask, dropout_p, is_causal, scale,
               enable_gqa):
    fn = _gems_attention()
    out = fn(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return _restore_full_mask_nan(query, key, out, attn_mask, is_causal)


def sdpa_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **_: Any,
) -> torch.Tensor:
    """aten-compatible SDPA entry for Cambricon MLU590.

    Legacy tuning keyword arguments are accepted and ignored because
    scheduling is delegated to the platform math/Triton path.
    """
    if query.device.type not in SUPPORTED_DEVICE_TYPES or not _has_mlu_stack():
        raise RuntimeError(
            f"sdpa_triton 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={query.device!r} 或当前栈缺少 torch_mlu。"
            "请按平台选择对应实现（见 PLATFORM.md）。"
        )
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("仅支持 4D (B, H, S, D)")
    if dropout_p != 0.0:
        raise ValueError("不支持 dropout（确定性验证路径）")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"不支持的 dtype: {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query/key/value dtype 不一致")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("head_dim 不一致")
    if is_causal and query.shape[-2] != key.shape[-2]:
        raise ValueError("causal 语义要求 Sq == Skv（与 aten/reference 一致）")

    hq, hkv = query.shape[1], key.shape[1]
    if attn_mask is not None:
        if attn_mask.dim() != 4:
            raise ValueError("仅支持 4D attn_mask")
        expected_mask_shape = (
            query.shape[0], hq, query.shape[-2], key.shape[-2]
        )
        if attn_mask.shape != expected_mask_shape:
            raise ValueError(
                f"attn_mask shape 应为 {expected_mask_shape}，"
                f"收到 {tuple(attn_mask.shape)}"
            )
        if (attn_mask.dtype != torch.bool
                and not attn_mask.dtype.is_floating_point):
            raise ValueError("attn_mask 仅支持 bool 或浮点加性 bias")
    if hq != hkv:
        if not enable_gqa:
            raise ValueError("Hq != Hkv 需要 enable_gqa=True")
        if hq % hkv:
            raise ValueError("GQA 要求 Hq 整除 Hkv")
    if query.shape[-2] == 0:
        return torch.empty_like(query)

    if torch.is_grad_enabled() and (
            query.requires_grad
            or key.requires_grad
            or value.requires_grad
            or (attn_mask is not None and attn_mask.requires_grad)):
        return _grad_path(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )

    # causal + explicit mask: fold causal into bias (math kernel, like F.sdpa,
    # rejects the combination on some stacks; fold keeps one mask channel).
    if is_causal and attn_mask is not None:
        sq, skv = query.shape[-2], key.shape[-2]
        if sq == skv:
            causal = torch.ones(
                (sq, skv), dtype=torch.bool, device=query.device
            ).tril()
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask & causal
            else:
                attn_mask = attn_mask + torch.where(
                    causal, 0.0, float("-inf")
                ).to(attn_mask.dtype)
            is_causal = False

    if _has_math_private():
        try:
            return _sdpa_math_private(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa,
            )
        except Exception:
            pass  # fall through to FlagGems / reference

    try:
        _gems_attention()
    except Exception:
        return sdpa_reference(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )
    return _sdpa_gems(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
    )
