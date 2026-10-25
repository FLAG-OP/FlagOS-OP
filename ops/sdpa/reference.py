# sdpa 语义参考实现——判卷标准，先于 kernel 编写。
#
# 数学语义（与 aten::scaled_dot_product_attention 对齐）:
#   out = softmax(Q @ K^T * scale + mask_bias) @ V
#   - scale 缺省 1/sqrt(D)；D = q.shape[-1]
#   - is_causal: 左上 tril 遮蔽，作用在 (Sq, Skv)——方阵/非方阵同一规则。
#     （2026-10-25 PR36 条目 4: 原先断言 Sq == Skv，实测与 aten 不符——
#     CPU 与 PPU 原生 F.sdpa 对 Sq≠Skv causal 均按 top-tril 出结果，
#     探针 probes/thead_nonsq_causal_probe.json native 行与 CPU fp32 参考
#     同容差（fp32 ≤2.53e-7 / fp16 ≤5.62e-4 / bf16 ≤2.98e-3，与 efficient
#     逐案同值）；
#     与 ops/sdpa_math/reference.py:149 同规则。）
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
        causal = torch.ones(Sq, Skv, dtype=torch.bool, device=q.device)
        causal = torch.tril(causal)          # 非方阵同为左上 tril
        scores = scores.masked_fill(~causal, _NEG_INF)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, _NEG_INF)
        else:
            scores = scores + attn_mask.float()

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.to(query.dtype)
