# P800 fast path for the exact sdpa_math output contract.
#
# The op must return both O and the full probability matrix P.  A no-P
# FlashAttention call is therefore not a valid replacement.  This path keeps
# the contract by splitting the work at a different boundary:
#
#   1. aten::_scaled_dot_product_efficient_attention computes O and row LSE;
#   2. one half-precision QK^T materializes P as exp(scale*QK^T - LSE);
#   3. causal/false regions are explicitly zeroed.
#
# Vendor LSE removes the row max/sum reduction from P generation while the
# vendor kernel supplies its mature O schedule.  Unsupported semantic corners
# fall back to torch_level rather than changing the operator contract.
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


def _load_neighbor(module_name: str, path: Path):
    """Load a sibling with a unique module name in the shared perf runner.

    The repository perf registry imports several operators by file path in one
    process.  A plain ``from kernel.torch_level import ...`` can then resolve
    to another operator's already-cached ``kernel`` package (#10 class issue).
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # Package-style import
    from ..reference import _validate
    from .torch_level import sdpa_math_torch
except ImportError:  # Standalone/top-level-kernel import
    _reference_module = _load_neighbor(
        "sdpa_math_p800_reference",
        Path(__file__).resolve().parents[1] / "reference.py",
    )
    _torch_module = _load_neighbor(
        "sdpa_math_p800_torch_level",
        Path(__file__).with_name("torch_level.py"),
    )
    _validate = _reference_module._validate
    sdpa_math_torch = _torch_module.sdpa_math_torch


PLATFORM = "p800-kunlunxin"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # XMLIR exposes Kunlunxin XPU as CUDA.
_FAST_DTYPES = (torch.float16, torch.bfloat16)


def _has_xmlir() -> bool:
    if importlib.util.find_spec("torch_xmlir") is None:
        return False
    try:
        import torch_xmlir  # noqa: F401
        return True
    except Exception:
        return False


def _fast_eligible(query, key, value, attn_mask, dropout_p, is_causal,
                   dropout_mask, scale) -> bool:
    if not _has_xmlir():
        return False
    if attn_mask is not None or dropout_p != 0.0 or dropout_mask is not None:
        return False
    if query.dtype not in _FAST_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.device.type != "cuda":
        return False
    if is_causal and query.shape[-2] != key.shape[-2]:
        return False
    if isinstance(scale, torch.Tensor):
        try:
            float(scale)
        except Exception:
            return False
    return True


def sdpa_math_p800_fast(
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
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, float,
]:
    """Return ``(output, attn_probs)`` using vendor O/LSE plus exact P.

    The fast path intentionally falls back to the fp32 torch composition when
    direct invocation could otherwise create a different autograd graph.  A1
    invokes it inside ``autograd.Function.forward`` (autograd disabled), where
    the specialized P800 backward can reuse the returned vendor state.
    """
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    build_graph_directly = (
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    if not _fast_eligible(query, key, value, attn_mask, dropout_p, is_causal,
                          dropout_mask, scale) or build_graph_directly:
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

    # This aten call is intentionally direct.  Calling F.sdpa would introduce a
    # Python-level backend-selection layer and could recurse after A1 hooking.
    result = torch.ops.aten._scaled_dot_product_efficient_attention(
        query, key, value, attn_bias=None, compute_log_sumexp=True,
        dropout_p=0.0, is_causal=is_causal, scale=actual_scale,
    )
    output, logsumexp = result[0], result[1]

    key_for_scores = key
    if query.shape[1] != key.shape[1]:
        key_for_scores = key.repeat_interleave(
            query.shape[1] // key.shape[1], dim=1)
    scores = torch.matmul(query, key_for_scores.transpose(-1, -2))
    scores.mul_(actual_scale).sub_(logsumexp.unsqueeze(-1)).exp_()

    if is_causal:
        sq, skv = query.shape[-2], key.shape[-2]
        causal = torch.ones(
            (sq, skv), dtype=torch.bool, device=query.device).tril()
        scores.masked_fill_(~causal, 0)
    if return_aux:
        # Private A1 path: keep the vendor state needed by the fused backward.
        # The two philox tensors are empty for dropout_p=0, but the aten
        # backward schema still requires them.
        return (output, scores, logsumexp, result[2], result[3],
                float(actual_scale))
    return output, scores
