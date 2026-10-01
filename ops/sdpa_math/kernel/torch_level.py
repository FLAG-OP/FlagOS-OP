# torch 级实现: 独立的 ATen 组合（不是 reference 的转发）。
# 价值: ①kernel 层的"第二判卷人"（同一语义、不同写法: softmax 守卫走
#       nan_to_num 而非 exp/sum 守卫）②A1 注册在无自研设备码平台上的
#       交付实现 ③数值问题三角定位的中间锚点。
#
# 约束: 只用原语算子（matmul/softmax/masked_fill），**不得**调用
# F.scaled_dot_product_attention 或被测算子本体——A1 注册后会递归
# （ops/sdpa 的 MLU 教训: 注册内转发原生 → RecursionError）。
from __future__ import annotations

import math

import torch

try:  # Package-style import: ops.sdpa_math.kernel.torch_level
    from ..reference import (CAUSAL_MASK_CONFLICT, GQA_NOT_DIVISIBLE,
                             _validate, make_dropout_mask)
except ImportError:  # Standalone import with OP_DIR on sys.path
    from reference import (CAUSAL_MASK_CONFLICT, GQA_NOT_DIVISIBLE,
                           _validate, make_dropout_mask)


def sdpa_math_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ATen 组合实现，返回 (out, attn_probs)。语义与 reference 一致。"""
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape

    # 与 reference 同规则: 实际输入 fp32 累积；fp64 输入保持 fp64（gradcheck）
    acc = torch.float64 if query.dtype == torch.float64 else torch.float32
    q = query.to(acc)
    k = key.to(acc)
    v = value.to(acc)

    if Hq != Hkv and enable_gqa:
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)

    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale)
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale

    if is_causal:
        # 与 reference 不同写法: where 直接构造加性 bias（广播到全形状）
        causal = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool,
                                       device=scores.device))
        bias = torch.where(causal, 0.0, float("-inf")).to(scores.dtype)
        scores = scores + bias

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(acc)

    # torch.softmax 对全 -inf 行给 NaN → nan_to_num 归 0（native 语义）
    attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)

    # dropout: 与 reference 同规则（两条 native 形态，见 reference 顶注）
    if dropout_p > 0.0:
        if dropout_mask is None:
            a = make_dropout_mask(attn.shape, dropout_p, attn.device)
            probs = attn * a
            out = torch.matmul(probs, v)
        else:
            keep = (dropout_mask != 0).to(attn.dtype).broadcast_to(
                attn.shape)
            probs = attn * keep
            out = torch.matmul(attn * (keep / (1.0 - dropout_p)), v)
    else:
        probs = attn
        out = torch.matmul(probs, v)
    return out.to(query.dtype), probs.to(query.dtype)
