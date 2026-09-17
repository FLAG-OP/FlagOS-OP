# sdpa 语义参考实现——判卷标准，先于 kernel 编写。
#
# 数学语义（与 aten::scaled_dot_product_attention 对齐）:
#   out = softmax(Q @ K^T * scale + mask_bias) @ V
#   - scale 缺省 1/sqrt(D)；D = q.shape[-1]
#   - is_causal: 仅支持 Sq == Skv 的下三角遮蔽（与 aten 语义一致）
#   - attn_mask: bool → False 处 -inf（遮蔽）；float → 加性 bias
#   - GQA: Hq != Hkv 时 K/V 按 Hq//Hkv 组广播
#   - 内部全程 fp32，最后 cast 回输入 dtype
#
# 独立性说明: 手写 matmul+softmax 组合而非调用 F.scaled_dot_product_attention，
# 使参考与被测实现（包括原生 SDPA）在实现上解耦——语义理解错误会在
# kernel 层精度测试中暴露，而不是被"参考恰好也是它"掩盖。
# 黄金生成时会与 CPU F.sdpa(fp32) 交叉互验（见 script/gen_golden.py）。
from __future__ import annotations

import math

import torch

_NEG_INF = float("-inf")


def sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """fp32 语义参考。支持 4D (B, H, Sq, D)；dropout 仅接受 0。"""
    assert dropout_p == 0.0, "参考实现不支持 dropout"
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, \
        "仅支持 4D (B, H, S, D)"

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, Dk = key.shape
    assert D == Dk == value.shape[-1], "head_dim 不一致"
    if Hq != Hkv:
        assert enable_gqa, "Hq != Hkv 需要 enable_gqa=True"
        assert Hq % Hkv == 0, "GQA 要求 Hq 整除 Hkv"

    q = query.float()
    k = key.float()
    v = value.float()

    # GQA: (B, Hkv, Skv, D) → (B, Hq, Skv, D)
    if Hq != Hkv:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale  # (B,H,Sq,Skv)

    if is_causal:
        assert Sq == Skv, "causal 语义要求 Sq == Skv（与 aten 一致）"
        causal = torch.ones(Sq, Skv, dtype=torch.bool, device=q.device)
        causal = torch.tril(causal)
        scores = scores.masked_fill(~causal, _NEG_INF)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, _NEG_INF)
        else:
            scores = scores + attn_mask.float()

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.to(query.dtype)
