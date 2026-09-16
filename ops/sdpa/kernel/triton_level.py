# Triton 级 SDPA 实现（flash-attention 风格 online-softmax two-pass）。
# ⚠ 平台: ascend910 专属——64×64 tile 上限 / exp2 负优化 / P-cast 等
# 绑定结论见 ../PLATFORM.md，移植前必读（CUDA 等平台多项结论相反）。
# wt <wangt635@ustc.edu.cn>
#
# 设计要点（均有实测/上游依据）:
#   1. 启动必须包 torch_device_fn.device 上下文（FlagOS-OP known-issues #11，
#      裸 Triton 在本栈首次正常、之后静默 no-op）
#   2. 不用 @triton.autotune（#15b，本栈会选出非法 num_warps=5）；
#      固定 BLOCK 配置 + 按 head_dim 分档
#   3. 尾块安全: K/V 遍历 masked load（other=-inf），exp 后贡献 0，
#      不污染 online 归约累加器（行级 online max/sum，区别于 #15a 的
#      块内 tl.sum 场景；实测见 test/kernel_level.py）
#   4. 语义对齐 aten::scaled_dot_product_attention:
#      4D (B,Hq,Sq,D)/(B,Hkv,Skv,D)、GQA 广播（Hq%Hkv==0）、causal
#      （Sq==Skv 下三角）、bool/float 4D attn_mask、fp32 内部累加、
#      输出 cast 回 q.dtype、scale 缺省 1/sqrt(D)
#   5. 完全确定性: 无 dropout、无原子操作——哨兵检查友好
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# ── 平台元数据（框架集成方用于程序化过滤；详见 ../PLATFORM.md）──
PLATFORM = "ascend910"                  # 本实现绑定平台
SUPPORTED_DEVICE_TYPES = ("npu",)       # 守卫放行的 torch device.type

_D_CONFIGS = {32: 32, 64: 64, 80: 128, 96: 128, 128: 128, 192: 256, 256: 256}

# BLOCK_M/BLOCK_N 固定分档（保守取值，UB 压力小；910 单核 UB 192KB）
def _blocks(head_dim: int) -> tuple[int, int]:
    if head_dim <= 64:
        return 64, 64
    if head_dim <= 128:
        return 64, 64
    return 32, 32


@triton.jit
def _sdpa_fwd_kernel(
    Q, K, V, O,
    sq, skv,
    scale,
    n_heads_q, gqa_rep,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    M_BOOL, HAS_BMASK,            # bool mask（遮蔽语义）
    M_FLOAT, HAS_FMASK,           # float mask（加性语义）
    stride_mb, stride_mh, stride_mm, stride_mn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_POW2: tl.constexpr, D_ACTUAL: tl.constexpr,
    INPUT_FP32: tl.constexpr,     # fp32 输入时 dot 用 ieee（默认 tf32 → 3e-4 级误差）
    LOAD_KT: tl.constexpr,        # 性能变体: K 直接按 (D, N) 布局加载，消 tl.trans
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_heads_q
    h = pid_bh % n_heads_q
    h_kv = h // gqa_rep            # GQA: Hq//Hkv 组共享

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # Sq 行
    offs_n = tl.arange(0, BLOCK_N)                        # Skv 列（块内）
    offs_d = tl.arange(0, D_POW2)                         # head_dim
    d_mask = offs_d < D_ACTUAL

    # Q 块: (BLOCK_M, D_POW2)
    q = tl.load(
        Q + b * stride_qb + h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < sq) & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # 行最大
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # exp 和
    acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)        # 输出累加

    # wt 2026-09-16-fix causal 循环截断: 下三角 M 块只需遍历到对角块
    # （实测 S=2048 causal 1.93x 加速; 编译器无法从运行时 where 推断
    # 循环边界收缩——原写法白算整个上三角区域）
    # # wt <wangt635@ustc.edu.cn>
    hi = tl.cdiv(skv, BLOCK_N)
    if IS_CAUSAL:
        hi = tl.minimum(hi, tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N))
    for start_n in range(0, hi):
        cols = start_n * BLOCK_N + offs_n
        n_mask = cols < skv
        # scores: (BLOCK_M, BLOCK_N) fp32
        # wt 2026-09-16-fix 性能变体 LOAD_KT: K 直接按 (D, N) 转置布局
        # 加载，消除 dot 前的 tl.trans（Ascend 上转置映射可能不进 cube）
        # # wt <wangt635@ustc.edu.cn>
        if LOAD_KT:
            kt = tl.load(
                K + b * stride_kb + h_kv * stride_kh
                + cols[None, :] * stride_kn + offs_d[:, None],
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        else:
            k = tl.load(
                K + b * stride_kb + h_kv * stride_kh
                + cols[:, None] * stride_kn + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            kt = tl.trans(k)
        # wt 2026-09-16-fix fp32 输入时 QK/PV 用 ieee 点积（默认 tf32 的
        # 10-bit 尾数会引入 ~3e-4 误差，extreme case 超容差）
        # # wt <wangt635@ustc.edu.cn>
        if INPUT_FP32:
            s = tl.dot(q, kt, out_dtype=tl.float32,
                       input_precision="ieee") * scale
        else:
            s = tl.dot(q, kt, out_dtype=tl.float32) * scale

        # 无效列（尾块）置 -inf → exp 后贡献 0
        s = tl.where(n_mask[None, :], s, float("-inf"))

        if IS_CAUSAL:
            # Sq==Skv 下三角: 行 m 只看列 n<=m
            s = tl.where(cols[None, :] <= offs_m[:, None], s, float("-inf"))

        if HAS_BMASK:
            bm = tl.load(
                M_BOOL + b * stride_mb + h * stride_mh
                + offs_m[:, None] * stride_mm + cols[None, :] * stride_mn,
                mask=(offs_m[:, None] < sq) & n_mask[None, :],
                other=0,
            )
            s = tl.where(bm != 0, s, float("-inf"))

        if HAS_FMASK:
            fm = tl.load(
                M_FLOAT + b * stride_mb + h * stride_mh
                + offs_m[:, None] * stride_mm + cols[None, :] * stride_mn,
                mask=(offs_m[:, None] < sq) & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            s = s + fm

        # online softmax 更新
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # 全 -inf 行（死行/全遮蔽）: m_new=-inf, exp(-inf-(-inf))=nan——
        # 限制 alpha: m_i==-inf 且 m_new==-inf 时取 1.0
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
        p = tl.exp(s - m_new[:, None])        # 尾块 -inf → 0
        p = tl.where(s == float("-inf"), 0.0, p)   # nan 行保险
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # V 块: (BLOCK_N, D_POW2)
        v = tl.load(
            V + b * stride_vb + h_kv * stride_vh
            + cols[:, None] * stride_vn + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # wt 2026-09-16-fix Ascend tl.dot 要求同 dtype: P cast 到 v.dtype
        # # wt <wangt635@ustc.edu.cn>
        if INPUT_FP32:
            acc = acc * alpha[:, None] + tl.dot(
                p.to(v.dtype), v, out_dtype=tl.float32,
                input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(
                p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    # 归一化: l=0（全遮蔽行，与 aten 一致输出 NaN）外正常除
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    # 布尔量做乘法把 NaN 保留: 全遮蔽行 acc=0, 0/1=0 ≠ aten 的 NaN。
    # 与参考对齐: 全遮蔽行输出 NaN（softmax(-inf...)=nan 语义）
    acc = tl.where(l_i[:, None] == 0.0, float("nan"), acc)

    tl.store(
        O + b * stride_ob + h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=(offs_m[:, None] < sq) & d_mask[None, :],
    )


def sdpa_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int = 4,
    num_stages: int | None = None,
    load_kt: bool = False,
) -> torch.Tensor:
    """aten::scaled_dot_product_attention 同签名 Triton 实现（4D）。"""
    # wt 2026-09-16-fix 设备守卫: 本 kernel 是 ascend910 绑定实现，
    # 跨平台误引（如 cuda tensor）在此显式拦截——防止静默用错平台
    # 配置（64×64 tile / exp 选择等）跑出慢或错的结果。
    # # wt <wangt635@ustc.edu.cn>
    if query.device.type not in SUPPORTED_DEVICE_TYPES:
        raise RuntimeError(
            f"sdpa_triton 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device.type={query.device.type!r}。"
            f"请按平台选择对应实现（见 PLATFORM.md），"
            f"或在 npu 设备上调用。")
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4, \
        "仅支持 4D (B, H, S, D)"
    assert dropout_p == 0.0, "不支持 dropout（确定性路径）"
    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, Dk = key.shape
    assert D == Dk == value.shape[-1], "head_dim 不一致"
    assert query.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        f"不支持的 dtype: {query.dtype}"

    if Hq != Hkv:
        assert enable_gqa, "Hq != Hkv 需要 enable_gqa=True"
        assert Hq % Hkv == 0, "GQA 要求 Hq 整除 Hkv"
    gqa_rep = Hq // Hkv

    if Sq == 0:
        return torch.empty_like(query)

    q, k, v = query, key, value
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    m_bool = m_float = None
    if attn_mask is not None:
        assert attn_mask.dim() == 4, "仅支持 4D attn_mask"
        if attn_mask.dtype == torch.bool:
            m_bool = attn_mask if attn_mask.is_contiguous() \
                else attn_mask.contiguous()
        else:
            m_float = attn_mask if attn_mask.is_contiguous() \
                else attn_mask.contiguous()

    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)
    d_pow2 = _D_CONFIGS.get(D) or triton.next_power_of_2(D)
    assert d_pow2 <= 512, f"head_dim={D} 超出支持范围"
    BLOCK_M, BLOCK_N = _blocks(D)
    # wt 2026-09-16-fix 性能实验入口: 显式参数覆盖默认分档（不改变默认行为）
    # # wt <wangt635@ustc.edu.cn>
    if block_m is not None:
        BLOCK_M = block_m
    if block_n is not None:
        BLOCK_N = block_n

    out = torch.empty_like(q)
    from flag_gems.runtime import torch_device_fn   # #11: 必须 device 上下文

    grid = (triton.cdiv(Sq, BLOCK_M), B * Hq)
    launch_kw = {"num_warps": num_warps}
    if num_stages is not None:
        launch_kw["num_stages"] = num_stages
    with torch_device_fn.device(q.device):
        _sdpa_fwd_kernel[grid](
            q, k, v, out,
            Sq, Skv, sm_scale,
            Hq, gqa_rep,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            m_bool if m_bool is not None else q,   # 空占位（HAS=0 不读）
            m_bool is not None,
            m_float if m_float is not None else q,
            m_float is not None,
            (m_bool if m_bool is not None else
             m_float if m_float is not None else q).stride(0),
            (m_bool if m_bool is not None else
             m_float if m_float is not None else q).stride(1),
            (m_bool if m_bool is not None else
             m_float if m_float is not None else q).stride(2),
            (m_bool if m_bool is not None else
             m_float if m_float is not None else q).stride(3),
            IS_CAUSAL=is_causal and Sq == Skv,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            D_POW2=d_pow2, D_ACTUAL=D,
            INPUT_FP32=(q.dtype == torch.float32),
            LOAD_KT=load_kt,
            **launch_kw,
        )
    return out
