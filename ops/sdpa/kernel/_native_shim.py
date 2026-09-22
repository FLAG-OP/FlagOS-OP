# 原生实现 shim（bench/精度对照用）: torch_npu 的 F.sdpa。
# wt 2026-09-16-fix NPU aten 拒绝 is_causal=True 与显式 attn_mask 并存
# （CPU 容忍、NPU 抛错）——本栈可跑的对照路径: causal 折叠进 float mask
# # wt <wangt635@ustc.edu.cn>
from __future__ import annotations

import torch


def sdpa_native(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
    if is_causal and attn_mask is not None:
        Sq, Skv = query.shape[-2], key.shape[-2]
        if Sq == Skv:
            causal = torch.tril(torch.ones(
                Sq, Skv, dtype=torch.bool, device=query.device))
            m = attn_mask
            if m.dtype == torch.bool:
                m = m & causal                    # 两个遮蔽取交
            else:
                # 加性折叠: 保持 mask dtype 与 query 一致（NPU aten 要求
                # attn_mask.dtype ∈ {bool, query.dtype}）
                m = m + torch.where(
                    causal, 0.0, float("-inf")).to(m.dtype)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=m, dropout_p=dropout_p,
                is_causal=False, scale=scale, enable_gqa=enable_gqa)
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


def _fold_causal_into_mask(query, key, attn_mask, is_causal):
    """causal 折叠辅助（auto_dispatch 直调 aten 用; auto 路径
    attn_mask 恒为 None，仅需处理 causal）。"""
    if not is_causal or attn_mask is not None:
        return attn_mask
    Sq, Skv = query.shape[-2], key.shape[-2]
    if Sq != Skv:
        return None  # 非方阵 causal 交由 aten 自身语义
    causal = torch.tril(torch.ones(
        Sq, Skv, dtype=torch.bool, device=query.device))
    return causal
