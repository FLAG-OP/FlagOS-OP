# Triton 级 SDPA math 后端实现（返回注意力概率图 P）。
# ⚠ 平台: ascend910 绑定——64×64 tile / ieee dot / device 上下文等
# 结论继承 ops/sdpa PLATFORM.md 的 Ascend 绑定清单，移植前必读。
#
# 与 flash 风格 online-softmax（ops/sdpa/kernel/backends/ascend910.py）的
# 根本差别: 本算子**必须返回 P = softmax(S)**（(B,Hq,Sq,Skv) 全矩阵），
# "只留行归约量"的省显存策略不可用。但 P 与 O 可以在**同一个 kernel**里
# 一次算完——P tile 在片上直接喂给 PV，不落地再读回（单 launch 融合）:
#   _probs_kernel(HAS_PV=1)  ← 默认路径（dropout_p=0，A1 注册恒为 0）
#     pass1 流式算行 max / 行 exp 和（online 统计量）
#     pass2 重算 scores → p=exp(S-max)/sum → 写 P + 同 tile 累加 O=P@V
#     causal 尾部整块（对角线以上）在本 kernel 内补写 0 → 无 torch.zeros
#   （两遍共用 _score_block，保证 scores 逐位一致，否则归一化出错）
#   dropout_p>0 无法先融合（掩码必须先作用在 P 上）→ 退回三段旧路径:
#     _probs_kernel(HAS_PV=0) → ATen 逐元素乘 → _pv_kernel
#   收益: launch 3→1（本栈单次启动实测 ~97µs）+ 省掉 P 的整块读回；
#   注意 O 与旧两段式逐位一致（p 先 cast 回 P.dtype 再 dot，等价于读回 fp16 P）
# 三处平台硬约束（继承 known-issues）:
#   #11 启动必须包 torch_device_fn.device 上下文（否则后续静默 no-op）
#   #15b 不用 @triton.autotune（本栈会选出非法 num_warps=5）
#   fp32 输入必须 input_precision="ieee"（默认 tf32 10-bit 尾数 ~3e-4 误差）
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

try:  # Package-style import
    from ..reference import _validate, make_dropout_mask
except ImportError:  # Standalone import
    from reference import _validate, make_dropout_mask

# ── 平台元数据（框架集成方用于程序化过滤）──
PLATFORM = "ascend910"
SUPPORTED_DEVICE_TYPES = ("npu",)


def _blocks(head_dim: int) -> tuple[int, int]:
    """固定分档（910 单核 UB 192KB；无 autotune，见 #15b）。"""
    if head_dim <= 128:
        return 64, 64
    return 32, 32


def _d1(x: torch.Tensor) -> torch.Tensor:
    """kernel 用 offs_d 直寻址 D 维（不传 stride_d）→ 只需 stride(-1)==1。
    满足即直接用原视图（transposed k/v 免拷贝——e2e 形状实测单次
    contiguous ~14µs，decode 三次共 ~41µs），否则物化。"""
    return x if x.stride(-1) == 1 else x.contiguous()


# 单 kernel 融合的适用范围（全部实测，见 reports/performance.md §6/§9）:
#   真正的判据是**实际处理的 score tile 数**（causal 决定每行块的截断上界），
#   而不是原始 Sq×Skv。交叉点按 e2e 口径（步内流水，等效吞吐口径）实测:
#     <=55 tiles 融合快: decode_cache 17-18 tiles 强制融合 -15%;
#     decode_re causal 36-55 tiles 强制融合 ~9%（304 vs 326ms）;
#     64+ tiles 两段式快（1x4096 64 tiles 250 vs 222µs; prefill 1024²
#     136 tiles 0.684 vs 0.522ms）→ 阈值 56 tiles。
#   单次调用同步的延迟口径在 32-55 tiles 会低估融合（两段式多 1-2 次
#   kernel 启动，本栈 ~97µs/次，流水下才兑现）→ 阈值以 e2e 实测为准。
#   门禁形状（1/3/136 tiles）不受阈值影响。
# 环境变量可强制: SDPA_MATH_FUSED=on/off（默认 auto，仅用于实验与归因）
_FUSED_MAX_TILES = 56


def _use_fused(sq: int, skv: int, head_dim: int, is_causal: bool,
               dropout_p: float) -> bool:
    if dropout_p != 0.0:
        return False                      # 掩码须先作用在 P 上，见文件头
    mode = os.environ.get("SDPA_MATH_FUSED", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    bm, bn = _blocks(head_dim)
    c = lambda x, y: (x + y - 1) // y          # noqa: E731
    if is_causal:
        tiles = sum(min(c(skv, bn), c(m0 + bm, bn))
                    for m0 in range(0, sq, bm))
    else:
        tiles = c(sq, bm) * c(skv, bn)
    return tiles <= _FUSED_MAX_TILES


@triton.jit
def _score_block(
    q,                              # (BLOCK_M, D)  fp32/half
    k,                              # (BLOCK_N, D)
    cols, offs_m, sq, skv, sm_scale,
    b, h,
    M_PTR, FM_PTR,
    stride_mb, stride_mh, stride_mm, stride_mn,
    HAS_BMASK: tl.constexpr,
    HAS_FMASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    """s = scale·q@kᵀ，再叠 causal / bool mask / float mask。
    返回 (BLOCK_M, BLOCK_N) fp32；无效列与遮蔽位置一律 -inf。"""
    n_mask = cols < skv
    if INPUT_FP32:
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32,
                   input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    s = s * sm_scale
    s = tl.where(n_mask[None, :], s, float("-inf"))      # 尾块无效列

    if IS_CAUSAL:
        # tril(diagonal=0): 行 m 只看列 n<=m（方阵/非方阵同规则，native 实测）
        s = tl.where(cols[None, :] <= offs_m[:, None], s, float("-inf"))

    if HAS_BMASK:
        bm = tl.load(
            M_PTR + b * stride_mb + h * stride_mh
            + offs_m[:, None] * stride_mm + cols[None, :] * stride_mn,
            mask=(offs_m[:, None] < sq) & n_mask[None, :],
            other=0,
        )
        s = tl.where(bm != 0, s, float("-inf"))

    if HAS_FMASK:
        fm = tl.load(
            FM_PTR + b * stride_mb + h * stride_mh
            + offs_m[:, None] * stride_mm + cols[None, :] * stride_mn,
            mask=(offs_m[:, None] < sq) & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        s = s + fm
    return s


@triton.jit
def _probs_kernel(
    Q, K, P,
    V, O,
    sq, skv, sm_scale,
    n_heads_q, gqa_rep,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_pb, stride_ph, stride_pm, stride_pn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    M_PTR, FM_PTR,
    stride_mb, stride_mh, stride_mm, stride_mn,
    HAS_BMASK: tl.constexpr,
    HAS_FMASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_PV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_POW2: tl.constexpr, D_ACTUAL: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_heads_q
    h = pid_bh % n_heads_q
    h_kv = h // gqa_rep                       # GQA: Hq//Hkv 组共享 K/V

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_POW2)
    d_mask = offs_d < D_ACTUAL
    row_ok = offs_m < sq
    q = tl.load(
        Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm
        + offs_d[None, :],
        mask=row_ok[:, None] & d_mask[None, :],
        other=0.0,
    )

    # causal 下超出对角的列恒为 -inf → 不参与统计；写出的 P 在该处为 0，
    # hi 之后的整块（对角线以上）由 pass2 结尾的补写循环填 0
    hi = tl.cdiv(skv, BLOCK_N)
    if IS_CAUSAL:
        hi = tl.minimum(hi, tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N))

    # ── pass1: 流式行 max / 行 exp 和 ──
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    for start_n in range(0, hi):
        cols = start_n * BLOCK_N + offs_n
        k = tl.load(
            K + b * stride_kb + h_kv * stride_kh + cols[:, None] * stride_kn
            + offs_d[None, :],
            mask=(cols < skv)[:, None] & d_mask[None, :],
            other=0.0,
        )
        s = _score_block(
            q, k, cols, offs_m, sq, skv, sm_scale, b, h,
            M_PTR, FM_PTR, stride_mb, stride_mh, stride_mm, stride_mn,
            HAS_BMASK=HAS_BMASK, HAS_FMASK=HAS_FMASK,
            IS_CAUSAL=IS_CAUSAL, INPUT_FP32=INPUT_FP32,
        )
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # 全 -inf 行（全遮蔽）: m_new=-inf 时 alpha 取 1，避免 exp(NaN)
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
        p_blk = tl.exp(s - m_new[:, None])
        p_blk = tl.where(s == float("-inf"), 0.0, p_blk)
        l_i = l_i * alpha + tl.sum(p_blk, axis=1)
        m_i = m_new

    # ── pass2: 重算 scores → 归一化写 P；HAS_PV 时同 tile 直接累加 O ──
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)   # 全遮蔽行: 分母兜底 → P=0
    if HAS_PV:
        acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)
    for start_n in range(0, hi):
        cols = start_n * BLOCK_N + offs_n
        k = tl.load(
            K + b * stride_kb + h_kv * stride_kh + cols[:, None] * stride_kn
            + offs_d[None, :],
            mask=(cols < skv)[:, None] & d_mask[None, :],
            other=0.0,
        )
        s = _score_block(
            q, k, cols, offs_m, sq, skv, sm_scale, b, h,
            M_PTR, FM_PTR, stride_mb, stride_mh, stride_mm, stride_mn,
            HAS_BMASK=HAS_BMASK, HAS_FMASK=HAS_FMASK,
            IS_CAUSAL=IS_CAUSAL, INPUT_FP32=INPUT_FP32,
        )
        p_blk = tl.exp(s - m_i[:, None]) / l_safe[:, None]
        p_blk = tl.where(s == float("-inf"), 0.0, p_blk)
        # 先 cast 回 P.dtype 再存/再 dot —— 等价于旧两段式的"落地读回"，
        # O 与旧路径逐位一致（golden 与 kernel 层回归据此校验）
        p_out = p_blk.to(P.dtype.element_ty)
        tl.store(
            P + b * stride_pb + h * stride_ph + offs_m[:, None] * stride_pm
            + cols[None, :] * stride_pn,
            p_out,
            mask=row_ok[:, None] & (cols < skv)[None, :],
        )
        if HAS_PV:
            v = tl.load(
                V + b * stride_vb + h_kv * stride_vh + cols[:, None] * stride_vn
                + offs_d[None, :],
                mask=(cols < skv)[:, None] & d_mask[None, :],
                other=0.0,
            )
            if INPUT_FP32:
                acc += tl.dot(p_out, v, out_dtype=tl.float32,
                              input_precision="ieee")
            else:
                acc += tl.dot(p_out, v, out_dtype=tl.float32)

    # causal 尾部补写: hi 之后整块列全在对角线以上 → P=0（原为独立的
    # torch.zeros 预填，现挪进本 kernel，省一次设备启动）
    if IS_CAUSAL:
        for start_n in range(hi, tl.cdiv(skv, BLOCK_N)):
            cols = start_n * BLOCK_N + offs_n
            tl.store(
                P + b * stride_pb + h * stride_ph + offs_m[:, None] * stride_pm
                + cols[None, :] * stride_pn,
                tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32).to(
                    P.dtype.element_ty),
                mask=row_ok[:, None] & (cols < skv)[None, :],
            )

    if HAS_PV:
        tl.store(
            O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om
            + offs_d[None, :],
            acc.to(O.dtype.element_ty),
            mask=row_ok[:, None] & d_mask[None, :],
        )


@triton.jit
def _pv_kernel(
    P, V, O,
    sq, skv,
    n_heads_q, gqa_rep,
    stride_pb, stride_ph, stride_pm, stride_pn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_POW2: tl.constexpr, D_ACTUAL: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    """O = P @ V（fp32 累加，输出 cast 回 q.dtype）。"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_heads_q
    h = pid_bh % n_heads_q
    h_kv = h // gqa_rep

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_POW2)
    d_mask = offs_d < D_ACTUAL
    row_ok = offs_m < sq

    acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)
    hi = tl.cdiv(skv, BLOCK_N)
    if IS_CAUSAL:
        # P 的被截断区域是预分配的 0，跳过它们是精确的
        hi = tl.minimum(hi, tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N))
    for start_n in range(0, hi):
        cols = start_n * BLOCK_N + offs_n
        n_mask = cols < skv
        p = tl.load(
            P + b * stride_pb + h * stride_ph + offs_m[:, None] * stride_pm
            + cols[None, :] * stride_pn,
            mask=row_ok[:, None] & n_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            V + b * stride_vb + h_kv * stride_vh + cols[:, None] * stride_vn
            + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if INPUT_FP32:
            acc += tl.dot(p, v, out_dtype=tl.float32,
                          input_precision="ieee")
        else:
            acc += tl.dot(p, v, out_dtype=tl.float32)

    tl.store(
        O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om
        + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=row_ok[:, None] & d_mask[None, :],
    )


def sdpa_math_triton(
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
    """aten::_scaled_dot_product_attention_math 同签名 Triton 实现（4D）。
    返回 (out, attn_probs)，两者 dtype = query.dtype。"""
    # 设备守卫: 本 kernel 是 ascend910 绑定实现，跨平台误引在此显式拦截
    # （防止用错平台 tile/dot 配置跑出慢或错的结果，见 PLATFORM.md）
    if query.device.type not in SUPPORTED_DEVICE_TYPES:
        raise RuntimeError(
            f"sdpa_math_triton 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device.type={query.device.type!r}。"
            f"请改用 kernel/torch_level.py 或在 npu 设备上调用。")
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, Dk = key.shape
    if value.shape[1] != Hkv:
        raise RuntimeError("K/V head 数不一致")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise RuntimeError(
            f"q/k/v dtype 必须一致: {query.dtype}/{key.dtype}/{value.dtype}")
    if not (Hq == Hkv or enable_gqa or Hkv == 1):
        raise RuntimeError(
            f"Hq={Hq} Hkv={Hkv} 且未开 enable_gqa 的头数广播形态 "
            f"自研 Triton 不支持（torch 级实现可用）")

    gqa_rep = max(Hq // Hkv, 1)
    sm_scale = scale if scale is not None else 1.0 / math.sqrt(D)
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale)

    q = _d1(query)
    k = _d1(key)
    v = _d1(value)

    m_ptr = fm_ptr = q
    stride_mb = stride_mh = stride_mm = stride_mn = 0
    HAS_BMASK = HAS_FMASK = False
    if attn_mask is not None:
        # 广播到 (B,Hq,Sq,Skv)（0 步长视图即可，kernel 按 stride 读，不物化）
        m4 = attn_mask.broadcast_to((B, Hq, Sq, Skv))
        stride_mb, stride_mh, stride_mm, stride_mn = m4.stride()
        if m4.dtype == torch.bool:
            m_ptr, HAS_BMASK = m4, True
        else:
            fm_ptr, HAS_FMASK = m4, True

    if Sq == 0 or Skv == 0:
        return (torch.empty_like(query),
                torch.empty((B, Hq, Sq, Skv), dtype=query.dtype,
                            device=query.device))

    # causal 尾部的 0 由 _probs_kernel 内补写 → P 一律 empty（省一次
    # torch.zeros 的独立设备启动，本栈单次启动 ~97µs）。
    # P 与 O 合并为一次 empty + 两个 view（省一次分配/启动 ~6µs），
    # 同时保证 out 连续 → stride(-1)==1（kernel 对 O 直寻址 D 维）
    n_p = B * Hq * Sq * Skv
    buf = torch.empty(n_p + B * Hq * Sq * D, dtype=query.dtype,
                      device=query.device)
    probs = buf[:n_p].view(B, Hq, Sq, Skv)
    out = buf[n_p:].view(B, Hq, Sq, D)

    BLOCK_M, BLOCK_N = _blocks(D)
    d_pow2 = triton.next_power_of_2(D)
    assert d_pow2 <= 512, f"head_dim={D} 超出支持范围"
    input_fp32 = query.dtype == torch.float32

    # 选择: 小形状单 kernel 融合 / 大形状两段式（见 _use_fused 顶注）
    HAS_PV = _use_fused(Sq, Skv, D, is_causal, dropout_p)

    from flag_gems.runtime import torch_device_fn   # #11: 必须 device 上下文
    grid = (triton.cdiv(Sq, BLOCK_M), B * Hq)
    with torch_device_fn.device(q.device):
        _probs_kernel[grid](
            q, k, probs,
            v, out,
            Sq, Skv, sm_scale,
            Hq, gqa_rep,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            probs.stride(0), probs.stride(1), probs.stride(2),
            probs.stride(3),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            m_ptr, fm_ptr,
            stride_mb, stride_mh, stride_mm, stride_mn,
            HAS_BMASK=HAS_BMASK, HAS_FMASK=HAS_FMASK,
            IS_CAUSAL=is_causal, HAS_PV=HAS_PV,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            D_POW2=d_pow2, D_ACTUAL=D,
            INPUT_FP32=input_fp32,
            num_warps=4,
        )

        # dropout（两条 native 规则，见 reference 顶注 / probes 实测）:
        #   随机:  a = keep/(1-p) → 返回的 P 与 PV 输入同用 a
        #   显式:  keep=(mask!=0) → 返回的 P **不**缩放，PV 输入再除 (1-p)
        #          （native: 显式路径 O = (P/(1-p))@V，P 与 O 不自洽）
        # 掩码由 wrapper 用 ATen 生成/展开——kernel 保持无 RNG、完全确定
        # （dropout_p>0 时 _use_fused 恒 False → 一定走下方 PV 段）
        probs_pv = probs
        if dropout_p > 0.0:
            if dropout_mask is None:
                a = make_dropout_mask(probs.shape, dropout_p, probs.device)
                probs = probs * a
                probs_pv = probs
            else:
                keep = (dropout_mask != 0).to(probs.dtype).broadcast_to(
                    probs.shape)
                probs = probs * keep
                probs_pv = probs / (1.0 - dropout_p)

        if not HAS_PV:
            # 两段式路径（大形状 / dropout）: P 已落地，第二段读回做 PV；
            # 融合路径已在 _probs_kernel 内完成，此处不再启动
            _pv_kernel[grid](
                probs_pv, v, out,
                Sq, Skv,
                Hq, gqa_rep,
                probs_pv.stride(0), probs_pv.stride(1), probs_pv.stride(2),
                probs_pv.stride(3),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                IS_CAUSAL=is_causal,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                D_POW2=d_pow2, D_ACTUAL=D,
                INPUT_FP32=input_fp32,
                num_warps=4,
            )
    return out, probs
