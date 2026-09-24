"""Cambricon MLU590 SDPA backend.

Engineering choice
------------------
Native ``F.sdpa`` on MLU dispatches to
``aten::_scaled_dot_product_fused_attention_overrideable`` (CNNL FlashAttention
v2).  The earlier math-private primary path was an unfused bmm+softmax
composite and ran 0.11-0.55x native.  efficient/flash/cudnn aten private ops
still fall back to CPU and stay unavailable.

Primary path: fused overrideable — same kernel family as native, typically
~1.0x (often bitwise-equal).  Optional fast path: ``torch_mlu_ops
.flash_attention`` (CNNL ScaledDotProductAttn_v7) for eligible half-precision
shapes measured 1.3-2.7x native.  bool masks become additive ``-inf`` bias;
causal+mask folds into the mask; fully masked rows restore NaN.  Fallbacks:
math private → FlagGems → fp32 reference.  Differentiable direct calls
borrow the A1 math backward.
"""
from __future__ import annotations

import importlib.util
import os
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


def _has_fused_overrideable() -> bool:
    try:
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable
        return True
    except Exception:
        return False


def _has_math_private() -> bool:
    try:
        torch.ops.aten._scaled_dot_product_attention_math
        return True
    except Exception:
        return False


def _tmo_flash_attention():
    if importlib.util.find_spec("torch_mlu_ops") is None:
        raise RuntimeError("torch_mlu_ops not installed")
    import torch_mlu_ops as tmo
    return tmo.flash_attention


def _gems_attention():
    from flag_gems.runtime.backend._cambricon.ops.attention import (
        scaled_dot_product_attention_forward,
    )
    return scaled_dot_product_attention_forward


def _bool_to_additive(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Fused/math kernels mishandle bool on this stack; -inf bias is exact."""
    bias = torch.zeros_like(attn_mask, dtype=dtype)
    return bias.masked_fill(~attn_mask, float("-inf"))


def _to_bias(attn_mask: torch.Tensor | None, dtype: torch.dtype):
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return _bool_to_additive(attn_mask, dtype)
    if attn_mask.dtype != dtype:
        return attn_mask.to(dtype)
    return attn_mask


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


def _expand_gqa(query, key, value):
    """Fused overrideable has no enable_gqa; broadcast KV heads to Hq."""
    hq, hkv = query.shape[1], key.shape[1]
    if hq == hkv:
        return query, key, value
    rep = hq // hkv
    key = key.repeat_interleave(rep, dim=1)
    value = value.repeat_interleave(rep, dim=1)
    return query, key, value


def _grad_path(query, key, value, attn_mask, dropout_p, is_causal,
               scale, enable_gqa):
    """Route differentiable direct calls through A1 math backward."""
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


def _sdpa_fused_overrideable(query, key, value, attn_mask, dropout_p,
                             is_causal, scale, enable_gqa):
    """P0 primary: CNNL FlashAttention via aten fused overrideable."""
    q, k, v = _expand_gqa(query, key, value)
    bias = _to_bias(attn_mask, query.dtype)
    out = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        q, k, v,
        attn_bias=bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=scale,
    )[0]
    return _restore_full_mask_nan(query, key, out, attn_mask, is_causal)


def _tmo_eligible(query, key, value, attn_mask, is_causal) -> bool:
    """Fast path: half/bf16 only (fp32 stays on overrideable for golden).

    Disable with SDPA_MLU_TMO=0 to force the fused-overrideable path.
    """
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    return os.environ.get("SDPA_MLU_TMO", "1") != "0"


def _sdpa_tmo(query, key, value, attn_mask, dropout_p, is_causal, scale,
              enable_gqa):
    """P1 fast path: torch_mlu_ops.flash_attention (BSHD + BHSD bias)."""
    flash = _tmo_flash_attention()
    hq, hkv = query.shape[1], key.shape[1]
    sq, skv = query.shape[-2], key.shape[-2]
    d = query.shape[-1]
    sm_scale = scale if scale is not None else d ** -0.5

    qs = query.transpose(1, 2).contiguous()          # B,S,Hq,D
    ks = key.transpose(1, 2).contiguous()            # B,S,Hkv,D
    vs = value.transpose(1, 2).contiguous()
    if hq != hkv:
        rep = hq // hkv
        # TMO wants equal head counts on BSHD layout: expand on head axis (dim=2).
        ks = ks.repeat_interleave(rep, dim=2).contiguous()
        vs = vs.repeat_interleave(rep, dim=2).contiguous()

    bias_bhsd = None
    if attn_mask is not None:
        b = _to_bias(attn_mask, query.dtype)
        # TMO attn_bias: (B, Hq, Sq, Skv) — already BHSD for our masks.
        bias_bhsd = b

    out = flash(
        qs, ks, vs,
        out=None,
        cu_seq_lens_q=None,
        cu_seq_lens_kv=None,
        alibi_slope=None,
        attn_bias=bias_bhsd,
        max_seq_len_q=sq,
        max_seq_len_kv=skv,
        softmax_scale=sm_scale,
        is_causal=is_causal,
        compute_dtype=torch.float32,
        return_lse=False,
        out_dtype=query.dtype,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    # B,S,H,D → B,H,S,D
    if out.dim() == 4 and out.shape[1] == sq and out.shape[2] == hq:
        out = out.transpose(1, 2).contiguous()
    return _restore_full_mask_nan(query, key, out, attn_mask, is_causal)


def _sdpa_math_private(query, key, value, attn_mask, dropout_p, is_causal,
                       scale, enable_gqa):
    bias = _to_bias(attn_mask, query.dtype)
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

    Routes: TMO FA (optional) → fused overrideable → math → FlagGems →
    reference.  Legacy tuning kwargs are accepted and ignored.
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

    # causal + explicit mask: fold into one mask channel (same as F.sdpa /
    # _native_shim; both reject the combination on some stacks).
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

    # P1: TMO fast path (half precision).
    if _tmo_eligible(query, key, value, attn_mask, is_causal):
        try:
            return _sdpa_tmo(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa,
            )
        except Exception:
            pass  # fall through to fused overrideable

    # P0: fused overrideable (native CNNL FA v2 path).
    if _has_fused_overrideable():
        try:
            return _sdpa_fused_overrideable(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa,
            )
        except Exception:
            pass

    if _has_math_private():
        try:
            return _sdpa_math_private(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                enable_gqa,
            )
        except Exception:
            pass

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
