import torch
import triton
import triton.language as tl
import math


@triton.jit
def _sdpa_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Mask_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_mask_m, stride_mask_n,
    seq_len_q, seq_len_k, head_dim,
    scale: tl.float32,
    dropout_p: tl.float32,
    seed,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Offset base pointers for this batch/head
    Q_ptr = Q_ptr + pid_bh * stride_qh
    K_ptr = K_ptr + pid_bh * stride_kh
    V_ptr = V_ptr + pid_bh * stride_vh
    Out_ptr = Out_ptr + pid_bh * stride_oh

    # Row offsets for this block of queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    # For online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block [BLOCK_M, BLOCK_K]
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    q = tl.load(
        Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    # Determine the range of K/V blocks to iterate
    if IS_CAUSAL:
        end_n = tl.minimum(seq_len_k, (pid_m + 1) * BLOCK_M)
    else:
        end_n = seq_len_k

    # Iterate over K/V blocks
    for start_n in range(0, tl.cdiv(seq_len_k, BLOCK_N) * BLOCK_N, BLOCK_N):
        # Early exit for causal
        if IS_CAUSAL:
            if start_n >= end_n:
                break

        cur_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_N, BLOCK_K]
        k_mask = (cur_offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        k = tl.load(
            K_ptr + cur_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        # Compute QK^T: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk = qk * scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Apply attention mask
        if HAS_MASK:
            mask_vals = tl.load(
                Mask_ptr + offs_m[:, None] * stride_mask_m + cur_offs_n[None, :] * stride_mask_n,
                mask=(offs_m[:, None] < seq_len_q) & (cur_offs_n[None, :] < seq_len_k),
                other=0.0,
            ).to(tl.float32)
            qk = qk + mask_vals

        # Mask out-of-bounds keys
        qk = tl.where(cur_offs_n[None, :] < seq_len_k, qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block [BLOCK_N, BLOCK_K]
        v_mask = (cur_offs_n[:, None] < seq_len_k) & (offs_k[None, :] < head_dim)
        v = tl.load(
            V_ptr + cur_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        # Apply dropout to attention weights
        if HAS_DROPOUT:
            p_keep = 1.0 - dropout_p
            rand_offs = pid_bh * seq_len_q * seq_len_k + offs_m[:, None] * seq_len_k + cur_offs_n[None, :]
            rand_vals = tl.rand(seed, rand_offs)
            dropout_mask = rand_vals > dropout_p
            p = tl.where(dropout_mask, p / p_keep, 0.0)

        # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, BLOCK_K]
        p = p.to(tl.float32)
        acc += tl.dot(p, v, allow_tf32=False)

        m_i = m_new

    # Finalize: divide by sum of exponentials
    acc = acc / l_i[:, None]

    # Store output
    out_mask = (offs_m[:, None] < seq_len_q) & (offs_k[None, :] < head_dim)
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
        acc,
        mask=out_mask,
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    assert query.dim() == 4
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ensure contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    orig_dtype = query.dtype

    # Output tensor
    output = torch.empty_like(query)

    # Choose block sizes based on head_dim
    BLOCK_K = triton.next_power_of_2(head_dim)
    if BLOCK_K < 16:
        BLOCK_K = 16

    BLOCK_M = 64
    BLOCK_N = 64

    if seq_len_q <= 32:
        BLOCK_M = 32
    if seq_len_k <= 32:
        BLOCK_N = 32

    BLOCK_M = max(BLOCK_M, 16)
    BLOCK_N = max(BLOCK_N, 16)

    # Grid
    grid = (triton.cdiv(seq_len_q, BLOCK_M), batch_size * num_heads)

    # Strides
    stride_qb, stride_qh, stride_qm, stride_qk = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kk = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vk = value.stride()
    stride_ob, stride_oh, stride_om, stride_ok = output.stride()

    assert stride_qb == num_heads * stride_qh, "Query tensor must be contiguous in batch/head dims"
    assert stride_kb == num_heads * stride_kh, "Key tensor must be contiguous in batch/head dims"
    assert stride_vb == num_heads * stride_vh, "Value tensor must be contiguous in batch/head dims"
    assert stride_ob == num_heads * stride_oh, "Output tensor must be contiguous in batch/head dims"

    HAS_MASK = attn_mask is not None
    HAS_DROPOUT = dropout_p > 0.0

    if HAS_MASK:
        attn_mask = attn_mask.contiguous()
        if attn_mask.dim() == 2:
            stride_mask_m, stride_mask_n = attn_mask.stride()
        elif attn_mask.dim() == 4:
            attn_mask = attn_mask.contiguous()
            stride_mask_m = attn_mask.stride(-2)
            stride_mask_n = attn_mask.stride(-1)
        else:
            stride_mask_m, stride_mask_n = attn_mask.stride()[-2:]
    else:
        stride_mask_m = 0
        stride_mask_n = 0

    seed = 0
    if HAS_DROPOUT:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()

    mask_ptr = attn_mask if HAS_MASK else query

    _sdpa_kernel[grid](
        query, key, value, output,
        mask_ptr,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_mask_m, stride_mask_n,
        seq_len_q, seq_len_k, head_dim,
        scale, dropout_p, seed,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=is_causal,
        HAS_DROPOUT=HAS_DROPOUT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output.to(orig_dtype)