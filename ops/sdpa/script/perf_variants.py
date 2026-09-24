# SDPA kernel 写法粒度实验: 单项变体 A/B，量化"粗粒度写法"的代价
# 变体（每项只改一处，均基于当前主 kernel）:
#   V4 causal 截断: 循环上界收缩到对角块（省 ~50% FLOPs @ causal）
#   V3 尾块分裂: 主循环免 mask，尾块单独 peeled
#   V2 exp2: scale 预折 log2(e)，用 exp2 指令
#   V1 LOAD_KT: K 直接按 (D,N) 转置布局加载（消 tl.trans）
#   V5 dot acc: rescale 融进 dot 累加器
#   COMBO: 全部合并
# 运行: python3 script/perf_variants.py
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

LOG2E = tl.constexpr(1.4426950408889634)


# ───────────────────────── 单项/组合 kernel（结构同主 kernel，改动点标注）─────────────────────────
@triton.jit
def _sdpa_var_kernel(
    Q, K, V, O,
    sq, skv, scale,
    n_heads_q, gqa_rep,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_POW2: tl.constexpr, D_ACTUAL: tl.constexpr,
    V_CAUSAL_TRUNC: tl.constexpr,   # V4
    V_TAIL_SPLIT: tl.constexpr,     # V3
    V_EXP2: tl.constexpr,           # V2
    V_LOAD_KT: tl.constexpr,        # V1
    V_DOT_ACC: tl.constexpr,        # V5
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_heads_q
    h = pid_bh % n_heads_q
    h_kv = h // gqa_rep

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_POW2)
    d_mask = offs_d < D_ACTUAL

    q = tl.load(
        Q + b * stride_qb + h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < sq) & d_mask[None, :], other=0.0)

    # V2: scale 预折进 Q（log2 域），循环内免乘
    # 注: fp16 q * fp32 标量会提升为 fp32——必须 cast 回原 dtype 再进 dot
    if V_EXP2:
        q = (q * (scale * LOG2E)).to(K.dtype.element_ty)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)

    # V4: causal 时循环上界收缩到对角块（下三角语义，Sq==Skv）
    hi = tl.cdiv(skv, BLOCK_N)
    if V_CAUSAL_TRUNC and IS_CAUSAL:
        hi = tl.minimum(hi, tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N))

    # V3: 主循环只跑完整块（免 n_mask）；尾块 peeled
    if V_TAIL_SPLIT:
        hi_main = skv // BLOCK_N
    else:
        hi_main = hi

    for start_n in range(0, hi_main):
        cols = start_n * BLOCK_N + offs_n
        if V_TAIL_SPLIT:
            # 完整块: skv 对齐，全部列有效——编译期免 mask
            if V_LOAD_KT:
                kt = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[None, :] * stride_kn + offs_d[:, None],
                    mask=d_mask[:, None], other=0.0)
            else:
                k = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[:, None] * stride_kn + offs_d[None, :],
                    mask=d_mask[None, :], other=0.0)
                kt = tl.trans(k)
            if V_EXP2:
                s = tl.dot(q, kt, out_dtype=tl.float32)
            else:
                s = tl.dot(q, kt, out_dtype=tl.float32) * scale
            if IS_CAUSAL:
                s = tl.where(cols[None, :] <= offs_m[:, None], s,
                             float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == float("-inf"), 1.0,
                             tl.exp(m_i - m_new))
            if V_EXP2:
                p = tl.math.exp2(s - m_new[:, None])
            else:
                p = tl.exp(s - m_new[:, None])
            p = tl.where(s == float("-inf"), 0.0, p)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(
                V + b * stride_vb + h_kv * stride_vh
                + cols[:, None] * stride_vn + offs_d[None, :],
                mask=d_mask[None, :], other=0.0)
            if V_DOT_ACC:
                acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None],
                             out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(
                    p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new
        else:
            n_mask = cols < skv
            if V_LOAD_KT:
                kt = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[None, :] * stride_kn + offs_d[:, None],
                    mask=d_mask[:, None] & n_mask[None, :], other=0.0)
            else:
                k = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[:, None] * stride_kn + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                kt = tl.trans(k)
            if V_EXP2:
                s = tl.dot(q, kt, out_dtype=tl.float32)
            else:
                s = tl.dot(q, kt, out_dtype=tl.float32) * scale
            s = tl.where(n_mask[None, :], s, float("-inf"))
            if IS_CAUSAL:
                s = tl.where(cols[None, :] <= offs_m[:, None], s,
                             float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == float("-inf"), 1.0,
                             tl.exp(m_i - m_new))
            if V_EXP2:
                p = tl.math.exp2(s - m_new[:, None])
            else:
                p = tl.exp(s - m_new[:, None])
            p = tl.where(s == float("-inf"), 0.0, p)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(
                V + b * stride_vb + h_kv * stride_vh
                + cols[:, None] * stride_vn + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            if V_DOT_ACC:
                acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None],
                             out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(
                    p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new

    # V3 尾块 peeled（仅 skv % BLOCK_N != 0 时有一个带 mask 的块）
    if V_TAIL_SPLIT:
        if hi_main * BLOCK_N < skv:
            cols = hi_main * BLOCK_N + offs_n
            n_mask = cols < skv
            if V_LOAD_KT:
                kt = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[None, :] * stride_kn + offs_d[:, None],
                    mask=d_mask[:, None] & n_mask[None, :], other=0.0)
            else:
                k = tl.load(
                    K + b * stride_kb + h_kv * stride_kh
                    + cols[:, None] * stride_kn + offs_d[None, :],
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                kt = tl.trans(k)
            if V_EXP2:
                s = tl.dot(q, kt, out_dtype=tl.float32)
            else:
                s = tl.dot(q, kt, out_dtype=tl.float32) * scale
            s = tl.where(n_mask[None, :], s, float("-inf"))
            if IS_CAUSAL:
                s = tl.where(cols[None, :] <= offs_m[:, None], s,
                             float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == float("-inf"), 1.0,
                             tl.exp(m_i - m_new))
            if V_EXP2:
                p = tl.math.exp2(s - m_new[:, None])
            else:
                p = tl.exp(s - m_new[:, None])
            p = tl.where(s == float("-inf"), 0.0, p)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(
                V + b * stride_vb + h_kv * stride_vh
                + cols[:, None] * stride_vn + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            if V_DOT_ACC:
                acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None],
                             out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.dot(
                    p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    acc = tl.where(l_i[:, None] == 0.0, float("nan"), acc)
    tl.store(
        O + b * stride_ob + h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=(offs_m[:, None] < sq) & d_mask[None, :])


def run_variant(q, k, v, causal, **kw):
    B, Hq, Sq, D = q.shape
    _, Hkv, Skv, _ = k.shape
    gqa_rep = Hq // Hkv
    d_pow2 = triton.next_power_of_2(D)
    out = torch.empty_like(q)
    from flag_gems.runtime import torch_device_fn
    grid = (triton.cdiv(Sq, 64), B * Hq)
    sm_scale = 1.0 / math.sqrt(D)
    with torch_device_fn.device(q.device):
        _sdpa_var_kernel[grid](
            q, k, v, out, Sq, Skv, sm_scale, Hq, gqa_rep,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            IS_CAUSAL=causal, BLOCK_M=64, BLOCK_N=64,
            D_POW2=d_pow2, D_ACTUAL=D, **kw)
    return out


def bench(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


VARIANTS = {
    "base":         {"V_CAUSAL_TRUNC": False, "V_TAIL_SPLIT": False,
                     "V_EXP2": False, "V_LOAD_KT": False, "V_DOT_ACC": False},
    "V4_causal":    {"V_CAUSAL_TRUNC": True, "V_TAIL_SPLIT": False,
                     "V_EXP2": False, "V_LOAD_KT": False, "V_DOT_ACC": False},
    "V3_tailsplit": {"V_CAUSAL_TRUNC": False, "V_TAIL_SPLIT": True,
                     "V_EXP2": False, "V_LOAD_KT": False, "V_DOT_ACC": False},
    "V2_exp2":      {"V_CAUSAL_TRUNC": False, "V_TAIL_SPLIT": False,
                     "V_EXP2": True, "V_LOAD_KT": False, "V_DOT_ACC": False},
    "V1_loadkt":    {"V_CAUSAL_TRUNC": False, "V_TAIL_SPLIT": False,
                     "V_EXP2": False, "V_LOAD_KT": True, "V_DOT_ACC": False},
    "V5_dotacc":    {"V_CAUSAL_TRUNC": False, "V_TAIL_SPLIT": False,
                     "V_EXP2": False, "V_LOAD_KT": False, "V_DOT_ACC": True,
                     "_skip": "Ascend dot acc 参数要求 lhs/rhs 同 dtype"},
    "COMBO":        {"V_CAUSAL_TRUNC": True, "V_TAIL_SPLIT": True,
                     "V_EXP2": True, "V_LOAD_KT": True, "V_DOT_ACC": False},
    "COMBO_noKT":   {"V_CAUSAL_TRUNC": True, "V_TAIL_SPLIT": True,
                     "V_EXP2": True, "V_LOAD_KT": False, "V_DOT_ACC": False},
}


def main():
    from reference import sdpa_reference

    dev = "npu:0"
    g = torch.Generator(device="cpu").manual_seed(0)

    # ── 数值正确性先行（fp16 S=256 causal + noncausal + 尾块 S=100）──
    print("== 数值验证（vs fp32 参考，fp16 容差 2e-2）==")
    for tag, S in [("causal", 256), ("noncausal", 256), ("tail", 100)]:
        causal = tag != "noncausal"
        q = (torch.randn(2, 4, S, 64, generator=g) * 0.5
             ).to(torch.float16).to(dev)
        k = (torch.randn(2, 4, S, 64, generator=g) * 0.5
             ).to(torch.float16).to(dev)
        v = (torch.randn(2, 4, S, 64, generator=g) * 0.5
             ).to(torch.float16).to(dev)
        ref = sdpa_reference(q, k, v, None, 0.0, causal, None, False).float()
        for name, kw in VARIANTS.items():
            if kw.get("_skip"):
                print(f"  [skip] {name}: {kw['_skip']}")
                continue
            if name == "base":
                from kernel.triton_level import sdpa_triton
                out = sdpa_triton(q, k, v, None, 0.0, causal, None, False)
            else:
                out = run_variant(q, k, v, causal, **kw)
            err = (out.float() - ref).abs().max().item()
            assert err < 2e-2, f"{name} {tag} err={err}"
        print(f"  {tag}: 全部 {len(VARIANTS)} 变体 PASS")

    # ── 性能 A/B（S=2048 D=128 H=16，causal 与 noncausal 分开）──
    for causal in (True, False):
        print(f"\n== 性能 @ S=2048 D=128 H=16 causal={causal} fp16 ==")
        q = (torch.randn(1, 16, 2048, 128, generator=g)
             ).to(torch.float16).to(dev)
        k = (torch.randn(1, 16, 2048, 128, generator=g)
             ).to(torch.float16).to(dev)
        v = (torch.randn(1, 16, 2048, 128, generator=g)
             ).to(torch.float16).to(dev)
        F = torch.nn.functional
        t_nat = bench(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=causal))
        print(f"{'variant':14s} {'ms':>8s} {'vs_base':>8s} {'vs_native':>9s}")
        base_ms = None
        for name, kw in VARIANTS.items():
            if kw.get("_skip"):
                print(f"{name:14s}  [skip] {kw['_skip'][:40]}")
                continue
            if name == "base":
                from kernel.triton_level import sdpa_triton
                fn = lambda: sdpa_triton(q, k, v, None, 0.0, causal,
                                         None, False)
            else:
                fn = lambda: run_variant(q, k, v, causal, **kw)
            try:
                t = bench(fn)
            except Exception as e:
                print(f"{name:14s}  FAIL {type(e).__name__} "
                      f"{str(e)[:50]}")
                continue
            if base_ms is None:
                base_ms = t
            print(f"{name:14s} {t:8.3f} {base_ms/t:7.2f}x "
                  f"{t/t_nat:8.2f}x")


if __name__ == "__main__":
    main()
