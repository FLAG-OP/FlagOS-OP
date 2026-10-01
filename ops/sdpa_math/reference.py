# sdpa_math 语义参考——判卷标准，先于 kernel 编写。
#
# 目标算子: aten::_scaled_dot_product_attention_math
#   schema (torch 2.10 实测):
#     _scaled_dot_product_attention_math(
#         Tensor query, Tensor key, Tensor value,
#         Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False,
#         Tensor? dropout_mask=None, *, float? scale=None,
#         bool enable_gqa=False) -> (Tensor, Tensor)
#   第二个返回值是**注意力概率图** P（不是 logsumexp），形状
#   (B, Hq, Sq, Skv)——这是它区别于 flash/efficient 后端的关键:
#   需要 P 的消费方（蒸馏/可解释性/注意力熵正则）只能走 math 后端。
#
# 数学语义（每条都以 native 实测为准，见 probes/native_semantics.py）:
#   S = scale · Q @ Kᵀ                        scale 缺省 1/sqrt(D)
#   S[i,j] = -inf 当 is_causal 且 j > i        tril(diagonal=0)，**方阵非方阵都按左上**
#   S = S ⊙ mask（bool: False→-inf）或 S + mask（float: 加性）
#   P = softmax(S)                            全 -inf 行 → 0（不是 NaN）
#   dropout（仅 dropout_p > 0；dropout_p = 0 时显式 dropout_mask 被忽略）:
#     · 显式 dropout_mask（native 实测，torch 2.10）:
#         keep = (dropout_mask != 0)     ← **只当 0/非零 keep 指示，幅值被忽略**
#         返回的 P = softmax ⊙ keep      ← 不带 1/(1-p) 缩放
#         O = (softmax ⊙ keep / (1-p)) @ V  ← 带缩放 → **O ≠ 返回的 P@V**
#     · 无显式掩码（随机，F.sdpa 唯一会走的形态）:
#         keep ~ Bernoulli(1-p)；P = softmax ⊙ keep/(1-p)；O = P @ V（自洽）
#   O = (dropout 后参与乘 V 的概率) @ V
#   返回 (O, P)，两者 dtype 均为 query.dtype；内部 fp32
#   （fp64 输入按 fp64 累积——仅供 gradcheck / 数值诊断，fp16/
#    bf16/fp32 等实际输入一律 fp32，与 native 一致）
#
# 注意: 上面两条 dropout 形态**互不相同**（显式掩码不缩放返回 P、但缩放 O），
# 是 native 的实测行为而非设计；照抄才能做 drop-in 替换。证据与数据见
# probes/native_semantics.py（五种掩码幅值 × 随机路径的因子表）。
#
# 独立性说明: 手写 matmul+softmax 组合，不调用 F.scaled_dot_product_attention
# 也不调用被测算子本体——语义理解错误会在 kernel 层精度测试暴露，而不是被
# "参考恰好也是它"掩盖。黄金生成时与 native math 后端交叉互验
# （script/gen_golden.py），互验输入是 native 的**真实调用形态**。
#
# 与 native 直调的**唯一**有意分歧（native 直调路径的怪癖，真实 F.sdpa
# 路径不会出现，实测记录见 probes/native_semantics.py）:
#   bool attn_mask: native 直调按 0/1 加性处理（True→+1.0）；
#   而 F.sdpa 的 dispatcher 在进本算子**之前**已把 bool 转成 float 加性
#   -inf 掩码（实测 min=-inf, max=0）。本实现按遮蔽语义处理 bool
#   （与 F.sdpa 一致、与直调分歧）——因为 bool 在真实调用路径里根本到不了这里。
# 其余语义（全 -inf 行→0、两条 dropout 规则、causal 左上 tril、GQA 头
# 校验、causal+mask 冲突报错）与 native 直调逐条一致。
from __future__ import annotations

import math

import torch

_NEG_INF = float("-inf")

CAUSAL_MASK_CONFLICT = (
    "_scaled_dot_product_attention: Explicit attn_mask should not be set "
    "when is_causal=True"
)
GQA_NOT_DIVISIBLE = (
    "Number of heads in key and value must divide the number of heads in "
    "query"
)


def make_dropout_mask(
    shape,
    dropout_p: float,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """随机路径的乘子 a = keep / (1-p)（与 native 随机路径同分布）:
    keep ~ Bernoulli(1-p)，a = keep/(1-p)（即 P 与 O 都乘 a）。
    显式 dropout_mask 走另一条 native 规则（keep=(mask!=0)，返回 P 不缩放、
    O 缩放），不能用本函数的输出冒充——两条路径在 reference/kernel 里分开。
    generator=None 走设备全局 RNG（与 torch.manual_seed 对齐）。
    """
    assert 0.0 <= dropout_p < 1.0, f"dropout_p 非法: {dropout_p}"
    keep = (torch.rand(shape, device=device, generator=generator) >= dropout_p)
    return keep.to(dtype) / (1.0 - dropout_p)


def _softmax01(scores: torch.Tensor) -> torch.Tensor:
    """softmax(-1)，全 -inf 行返回 0（native/F.sdpa 实测语义）。
    参考用 exp/sum 显式守卫；torch 级实现走 nan_to_num（互为印证）。"""
    row_max = scores.amax(dim=-1, keepdim=True)
    safe_max = torch.where(row_max == _NEG_INF, torch.zeros_like(row_max),
                           row_max)
    p = torch.exp(scores - safe_max)
    denom = p.sum(dim=-1, keepdim=True)
    p = p / torch.where(denom == 0.0, torch.ones_like(denom), denom)
    return torch.where(denom == 0.0, torch.zeros_like(p), p)


def _validate(query, key, value, attn_mask, is_causal, enable_gqa) -> None:
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, \
        "仅支持 4D (B, H, S, D)"
    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, Dk = key.shape
    assert D == Dk == value.shape[-1], "head_dim 不一致"
    assert value.shape[0] == query.shape[0], "batch 不一致"
    assert key.shape[1] == value.shape[1], "K/V head 数不一致"
    if is_causal and attn_mask is not None:
        raise RuntimeError(CAUSAL_MASK_CONFLICT)
    if enable_gqa and Hq != Hkv and (Hkv == 0 or Hq % Hkv != 0):
        raise RuntimeError(GQA_NOT_DIVISIBLE)


def sdpa_math_reference(
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
    """fp32 语义参考，返回 (out, attn_probs)。

    dropout_p > 0 时两条 native 规则（见文件顶注与 probes/native_semantics.py）:
      · dropout_mask 显式给出 → keep=(mask!=0)；返回 P 不缩放、O 缩放；
      · dropout_mask=None      → 随机 keep，P 与 O 同乘 keep/(1-p)。
    黄金数据一律用显式 dropout_mask（跨设备确定性）。
    """
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape

    acc = torch.float64 if query.dtype == torch.float64 else torch.float32
    q = query.to(acc)
    k = key.to(acc)
    v = value.to(acc)

    if Hq != Hkv and enable_gqa:            # GQA: (B,Hkv,S,D) → (B,Hq,S,D)
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)

    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale)
    scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale  # (B,H,Sq,Skv)

    if is_causal:                           # tril(diagonal=0)，方阵/非方阵同规则
        causal = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool,
                                       device=scores.device))
        scores = scores.masked_fill(~causal, _NEG_INF)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, _NEG_INF)
        else:
            scores = scores + attn_mask.to(acc)

    attn = _softmax01(scores)                      # softmax（dropout 前）

    if dropout_p > 0.0:
        if dropout_mask is None:
            # 随机路径: P 与 O 都乘 keep/(1-p)（自洽）
            a = make_dropout_mask(attn.shape, dropout_p, attn.device)
            probs = attn * a
            out = torch.matmul(probs, v)
        else:
            # 显式路径: mask 只当 0/非零 keep 指示（幅值忽略），
            # 返回的 P 不缩放、O 缩放 → O = (P/(1-p)) @ V
            keep = (dropout_mask != 0).to(attn.dtype).broadcast_to(
                attn.shape)
            probs = attn * keep
            out = torch.matmul(attn * (keep / (1.0 - dropout_p)), v)
    else:
        probs = attn                               # dropout_mask 此时被忽略
        out = torch.matmul(probs, v)
    return out.to(query.dtype), probs.to(query.dtype)
