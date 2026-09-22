# KernelGen LLM-Track 本地复现: 模拟 generate_kernel 输出的 Triton SDPA
#
# 协议: 按 flagos-skills/kernelgen-flagos 的参数结构单轮生成:
#   kernel_name: scaled_dot_product_attention
#   func_type:   attention  (复杂类, 官方提示成功率 ~60%)
#   arg_names:   [q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa]
#   flagos_wiki: Kernel pattern: flash-attention tiled two-pass online softmax
#                Use Triton for GPU parallelization
#                Memory: qkv contiguous row-major (B,H,S,D)
#                Memory access: K/V streamed by blocks along S_kv
#                fp32 accumulator for softmax stability
#
# 生成约束（对比公平性）: 单轮裸生成，不含我们开发中获得的平台知识
# （AutogradPrivateUse1 注册点 / dot 同 dtype / tf32-ieee / exp2 慢路径 /
#  UB 192KB tile 上限 / #11 device ctx）——这些属于"人工/迭代轮"才知道的信息
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _sdpa_llm_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    N_HEAD, seq_len, scale,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # 标准 flash-attention v2 教科书布局（生成模型的默认先验）
    start_m = tl.program_id(0)
    off_hbz = tl.program_id(1)
    off_h = off_hbz % N_HEAD
    off_b = off_hbz // N_HEAD

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(q_ptrs)

    n_block = tl.cdiv(seq_len, BLOCK_N)
    for start_n in range(0, n_block):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        # causal（简化: 方阵假设）
        mask = offs_m[:, None] >= (start_n * BLOCK_N + offs_n)[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty))


def sdpa_llm_generated(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """LLM 单轮生成的 SDPA（causal-only 简化版，无 mask/GQA 处理——
    生成模型对完整 aten 签名的典型覆盖水平）。"""
    B, H, S, D = q.shape
    assert k.shape == v.shape == q.shape, "生成版仅支持 MHA 方阵"
    assert attn_mask is None and not enable_gqa, "生成版未覆盖"
    assert dropout_p == 0.0
    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)

    o = torch.empty_like(q)
    grid = (triton.cdiv(S, 64), B * H)
    _sdpa_llm_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, S, sm_scale, sm_scale,
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=triton.next_power_of_2(D),
        num_warps=4,
    )
    return o
