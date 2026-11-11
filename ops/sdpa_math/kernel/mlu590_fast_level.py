"""sdpa_math MLU590 (Cambricon) fast path: vendor fused overrideable + exact-P.

实测结论（torch_mlu 1.34.1 / Neuware 4.7.6 / Driver 6.2.29）:
  - _scaled_dot_product_fused_attention_overrideable 返回 9-tuple，
    [0]=O, [1]=LSE(shape=(B,Hq,Sq), dtype=fp32)，实测无 pad
    (Sq=100 时 last dim=100)。
  - LSE 语义 = 标准 logsumexp: P = exp(scale·QKᵀ − LSE)，max_abs_err≈1.2e-4。
  - schema: (q,k,v, attn_bias=None, dropout_p=0., is_causal=False,
             return_debug_mask=False, *, scale=None)，无 enable_gqa。

平台绑定（见 PLATFORM.md §3b）:
  - attn_bias 形状 (B,Hq,Sq,Skv)，bool 必须先转 additive。
  - GQA/MQA 通过 repeat_interleave 展 KV（overrideable 无 enable_gqa）。
  - causal 且 Sq≠Skv 时 fused overrideable 语义不匹配 aten math → 回退
    torch_level（与 sdpa/kernel/backends/mlu590.py 同判）。
  - causal + attn_mask 折叠进单一 mask（对齐 mlu590.py 做法）。
  - 全遮蔽行: aten math 语义 out=0/P=0（不是 NaN），vendor 可能返回有限
    值 → 显式 masked_fill(0)。
  - fp32 与 dropout_p>0 回退 torch_level（fast path 只承接半精度）。

A1 集成: register._fwd 以 dropout_p=0 调用本函数；dropout 在 Function
层落地。direct requires_grad 建图回退 torch_level（同 p800/thead）。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


def _load_neighbor(module_name: str, path: Path):
    """Load a sibling with a unique module name (multi-operator perf runner #10)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # Package-style import
    from ..reference import _validate
    from .torch_level import sdpa_math_torch
except ImportError:  # Standalone/top-level-kernel import
    _reference_module = _load_neighbor(
        "sdpa_math_mlu590_reference",
        Path(__file__).resolve().parents[1] / "reference.py",
    )
    _torch_module = _load_neighbor(
        "sdpa_math_mlu590_torch_level",
        Path(__file__).with_name("torch_level.py"),
    )
    _validate = _reference_module._validate
    sdpa_math_torch = _torch_module.sdpa_math_torch


PLATFORM = "mlu590"
SUPPORTED_DEVICE_TYPES = ("mlu",)
_FAST_DTYPES = (torch.float16, torch.bfloat16)

_HAS_MLU_STACK: bool | None = None
_HAS_FUSED_OVERRIDEABLE: bool | None = None


def _has_mlu_stack() -> bool:
    global _HAS_MLU_STACK
    if _HAS_MLU_STACK is None:
        try:
            import torch_mlu  # noqa: F401
            _HAS_MLU_STACK = True
        except Exception:
            _HAS_MLU_STACK = False
    return _HAS_MLU_STACK


def _has_fused_overrideable() -> bool:
    global _HAS_FUSED_OVERRIDEABLE
    if _HAS_FUSED_OVERRIDEABLE is None:
        try:
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable
            _HAS_FUSED_OVERRIDEABLE = True
        except Exception:
            _HAS_FUSED_OVERRIDEABLE = False
    return _HAS_FUSED_OVERRIDEABLE


def _bool_to_additive(attn_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool mask → additive -inf bias，语义即遮蔽（与 F.sdpa 对齐）。"""
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


def _expand_gqa(query, key, value):
    """Fused overrideable has no enable_gqa; broadcast KV heads to Hq."""
    hq, hkv = query.shape[1], key.shape[1]
    if hq == hkv:
        return query, key, value
    rep = hq // hkv
    key = key.repeat_interleave(rep, dim=1)
    value = value.repeat_interleave(rep, dim=1)
    return query, key, value


def _mask_can_broadcast(mask_shape, target_shape) -> bool:
    if len(mask_shape) != 4:
        return False
    return all(m == 1 or m == t for m, t in zip(mask_shape, target_shape))


def _fast_eligible(query, key, value, attn_mask, dropout_p, is_causal,
                   dropout_mask, scale) -> bool:
    if not _has_mlu_stack() or not _has_fused_overrideable():
        return False
    if query.device.type not in SUPPORTED_DEVICE_TYPES:
        return False
    if query.dtype not in _FAST_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if dropout_p != 0.0 or dropout_mask is not None:
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    # fused overrideable 的 causal 语义要求方阵（对齐 mlu590.py）
    if is_causal and query.shape[-2] != key.shape[-2]:
        return False
    if attn_mask is not None:
        target = (
            query.shape[0], query.shape[1],
            query.shape[-2], key.shape[-2],
        )
        if not _mask_can_broadcast(tuple(attn_mask.shape), target):
            return False
        if not (attn_mask.dtype == torch.bool
                or attn_mask.dtype.is_floating_point):
            return False
    if isinstance(scale, torch.Tensor):
        try:
            float(scale)
        except Exception:
            return False
    return True


def _materialize_p(query, key, lse, is_causal, scale, attn_mask=None):
    """P = exp(scale·QKᵀ − LSE)，fp32 中间计算，mask/causal 显式置 0。"""
    sq = query.shape[-2]
    # 实测无 pad；防御性切片，兼容未来 vendor 变更
    if lse.shape[-1] > sq:
        lse = lse[..., :sq]

    scores = torch.matmul(query, key.transpose(-1, -2)).float()
    scores = scores * scale - lse.unsqueeze(-1)
    P = scores.exp()

    if is_causal:
        skv = key.shape[-2]
        causal = torch.ones((sq, skv), dtype=torch.bool,
                            device=P.device).tril()
        P = P.masked_fill(~causal, 0.0)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            P = P.masked_fill(~attn_mask, 0.0)
        else:
            P = P.masked_fill(torch.isneginf(attn_mask), 0.0)

    # 全遮蔽行 exp(-inf - (-inf)) = NaN → 归 0（aten math 语义）
    P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return P.to(query.dtype)


def _restore_full_mask_zero(query, key, out, attn_mask, is_causal):
    """aten math: 全遮蔽行 out=0。vendor 可能返回有限值 → 显式置 0。"""
    effective_false = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            effective_false = ~attn_mask
        else:
            effective_false = torch.isneginf(attn_mask)
    if is_causal:
        sq, skv = query.shape[-2], key.shape[-2]
        causal = torch.ones((sq, skv), dtype=torch.bool,
                            device=query.device).tril()
        effective_false = (
            ~causal if effective_false is None
            else (effective_false | ~causal)
        )
    if effective_false is None:
        return out
    all_masked = effective_false.all(dim=-1, keepdim=True)
    if not bool(all_masked.any()):
        return out
    return out.masked_fill(all_masked, 0.0)


def sdpa_math_mlu590_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: torch.Tensor | None = None,
    *,
    scale: float | torch.Tensor | None = None,
    enable_gqa: bool = False,
    return_aux: bool = False,
):
    """Return ``(output, attn_probs)`` via MLU fused overrideable + exact-P."""
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    build_graph_directly = (
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    if not _fast_eligible(
        query, key, value, attn_mask, dropout_p, is_causal,
        dropout_mask, scale,
    ) or build_graph_directly:
        result = sdpa_math_torch(
            query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
            scale=scale, enable_gqa=enable_gqa,
        )
        return (*result, None, None, None, None) if return_aux else result

    actual_scale = (
        scale if scale is not None
        else 1.0 / math.sqrt(query.shape[-1])
    )
    if isinstance(actual_scale, torch.Tensor):
        actual_scale = float(actual_scale)

    target_shape = (
        query.shape[0], query.shape[1],
        query.shape[-2], key.shape[-2],
    )
    if attn_mask is not None and tuple(attn_mask.shape) != target_shape:
        attn_mask = attn_mask.expand(target_shape).contiguous()

    # causal + mask 折叠进单一 mask（对齐 mlu590.py 做法）
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

    query_exp, key_exp, value_exp = _expand_gqa(query, key, value)
    bias = _to_bias(attn_mask, query.dtype)

    r = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
        query_exp, key_exp, value_exp,
        attn_bias=bias,
        dropout_p=0.0,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=actual_scale,
    )
    output, lse = r[0], r[1]

    probs = _materialize_p(
        query_exp, key_exp, lse, is_causal, actual_scale, attn_mask=attn_mask,
    )
    output = _restore_full_mask_zero(
        query_exp, key_exp, output, attn_mask, is_causal,
    )

    if return_aux:
        return (output, probs, lse, None, None, float(actual_scale))
    return output, probs
