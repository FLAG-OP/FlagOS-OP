#!/usr/bin/env python3
"""PR36 条目 4 取证: Sq != Skv 的 causal 在 thead-ppu 上到底能不能走 efficient。

审阅条目 4 称「T-Head SDPA 文档写明 Sq != Skv 的 causal 会转 efficient，
但当前 backend 仍直接拒绝」。本探针在跑卡前给出可判定的实测证据:

  1) 生产 facade 行为 —— kernel/triton_level.sdpa_triton 对非方阵 causal
     还拒不拒（整改前 18 例 ValueError + reference 断言 1 例，快照见
     probes/out/thead_nonsq_causal_probe_prefix.json）；
  2) 原生 F.sdpa 在 PPU 对非方阵 causal 的实际语义（native parity 口径）；
  3) flash 直调对非方阵 causal 是否生效（sdpa_math 侧记录 flash err≈2.0）;
  4) efficient 直调对非方阵 causal 能否出结果 + 精度（top-tril fp32 参考），
     含 LSE 32-pad 切回、GQA 扩 KV、_aligned_bias 非对齐 bias 三条联动约束;
  5) autograd: 厂商 efficient 反向 vs 参考；**5b 是「旧门回放」**——
     内联复刻整改前 register.py 的 `if ctx.is_causal and Sq == Skv` 门
     （该门已拆），所以修后重跑 5b 仍报 MISMATCH，用来说明为什么必须拆门；
     现行代码路径的梯度判卷见 test/op_level.py 第 6 步。

判卷口径同 script/check_accuracy（CPU fp32 参考 + 黄金 abs 容差:
fp32 1e-5 / fp16·bf16 2e-2）。causal 非方阵语义按 aten 实测的 **top-tril**
（tril(diagonal=0) 作用在 (Sq, Skv)）—— CPU F.sdpa 已实证该约定
（见探针 native/square 对照行与 sdpa_math/reference.py:149 同款）。

用法:
  PYTHONPATH=/opt/FlagOS-OP python3 probes/thead_nonsq_causal_probe.py \
      [--device cuda:6]

输出: 逐项 status/err 记录 + 结论字段。
  现行状态（每次重跑覆盖）: probes/out/thead_nonsq_causal_probe.json
  整改前取证快照（只读, 2026-10-25 PR36 条目 4 选型依据）:
      probes/out/thead_nonsq_causal_probe_prefix.json
      —— prod_facade 18 例 ValueError + reference 1 例 AssertionError
         （summary REJECT 19）、a1 旧门 MISMATCH 4.678
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}


def dtn(dt) -> str:
    return str(dt).split(".")[-1]


def _ref(q, k, v, mask, causal, scale):
    """CPU fp32 top-tril 参考（sdpa_reference 同式，去掉方阵断言）。"""
    acc = torch.float32
    qq, kk, vv = q.to(acc), k.to(acc), v.to(acc)
    if q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
    s = torch.matmul(qq, kk.transpose(-1, -2))
    s = s * (scale if scale is not None else q.shape[-1] ** -0.5)
    if causal:
        sq, skv = q.shape[-2], k.shape[-2]
        causal_tril = torch.ones(sq, skv, dtype=torch.bool,
                                 device=s.device).tril()
        s = s.masked_fill(~causal_tril, float("-inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            s = s.masked_fill(~mask, float("-inf"))
        else:
            s = s + mask.to(acc)
    p = torch.softmax(s, dim=-1)
    out = torch.matmul(p, vv)
    lse = torch.logsumexp(s, dim=-1)
    return out, lse


def _err(out, ref):
    """最大绝对误差。

    参考自身非有限的位置（如 bool mask 下的全遮蔽行）不参与比较——否则
    NaN 会把整块 diff 污染成非有限；但**参考有限而 out 非有限**是真错，
    记 inf → FAIL（旧行为 `nan[..., None].expand_as(diff)` 对同秩形状直接
    抛 expand 错，把 bool1x64 这类合法 NaN 行误记成 ERROR）。
    """
    of = torch.isfinite(out)
    rf = torch.isfinite(ref)
    diff = (out.float() - ref.float()).abs()
    both = of & rf
    extra = int((rf & ~of).sum())
    d = diff[both]
    if extra:
        return float("inf")
    return d.max().item() if d.numel() else 0.0


def _status(err, dt):
    return "PASS" if err < TOL[dtn(dt)] else "FAIL"


def rec(res, name, status, err=None, note="", **kw):
    item = {"check": name, "status": status}
    if err is not None:
        item["err"] = err
    if note:
        item["note"] = note
    item.update(kw)
    res.append(item)
    print(f"  [{status:7s}] {name:44s} "
          f"err={(('%.3e' % err) if err is not None else '-'):>9s} {note}")


def mk(B, Hq, Hkv, Sq, Skv, D, dt, dev, seed=42, mask=None, dist="normal"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = torch.randn
    if dist == "large":
        rnd = lambda *n: torch.rand(*n, generator=g) * 40.0  # noqa: E731
    q = (rnd(*((B, Hq, Sq, D)), generator=g) * 0.5).to(dt).to(dev)
    k = (rnd(*((B, Hkv, Skv, D)), generator=g) * 0.5).to(dt).to(dev)
    v = (rnd(*((B, Hkv, Skv, D)), generator=g) * 0.5).to(dt).to(dev)
    m = None
    if mask == "bool":
        m = (torch.rand(B, Hq, Sq, Skv, generator=g) > 0.3).to(dev)
    elif mask == "float":
        m = (torch.randn(B, Hq, Sq, Skv, generator=g) * 0.1).to(dt).to(dev)
    return q, k, v, m


def check_shapes(res, dev):
    """1/2/3/4: facade 是否仍拒绝、native、flash、efficient 精度。"""
    shapes = [
        ("sq128-skv128", 128, 128),
        ("sq4-skv8", 4, 8),
        ("sq8-skv4", 8, 4),
        ("sq1-skv64", 1, 64),
        ("sq100-skv197", 100, 197),
        ("sq333-skv197", 333, 197),
        ("sq64-skv1", 64, 1),
    ]
    dtypes = (torch.float16, torch.bfloat16, torch.float32)

    from kernel.triton_level import sdpa_triton

    for tag, Sq, Skv in shapes:
        for dt in dtypes:
            q, k, v, _ = mk(1, 4, 4, Sq, Skv, 64, dt, dev, seed=42)
            ref, ref_lse = _ref(q, k, v, None, True, None)

            # 1) 生产 facade（唯一判分口径）
            try:
                out = sdpa_triton(q, k, v, None, 0.0, True, None, False)
                e0 = _err(out, ref)
                rec(res, f"prod_facade/{tag}/{dtn(dt)}",
                    _status(e0, dt), e0)
            except Exception as e:
                rec(res, f"prod_facade/{tag}/{dtn(dt)}", "REJECT",
                    note=f"{type(e).__name__}: {str(e)[:110]}")

            # 2) 原生 F.sdpa（native parity 口径）
            try:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                e = _err(out, ref)
                rec(res, f"native/{tag}/{dtn(dt)}", _status(e, dt), e)
            except Exception as ex:
                rec(res, f"native/{tag}/{dtn(dt)}", "ERROR",
                    note=f"{type(ex).__name__}: {str(ex)[:110]}")

            # 3) flash 直调（maskless 快路径候选）
            try:
                out = torch.ops.aten._scaled_dot_product_flash_attention(
                    q, k, v, 0.0, True, False)[0]
                e = _err(out, ref)
                rec(res, f"flash/{tag}/{dtn(dt)}", _status(e, dt), e)
            except Exception as ex:
                rec(res, f"flash/{tag}/{dtn(dt)}", "ERROR",
                    note=f"{type(ex).__name__}: {str(ex)[:110]}")

            # 4) efficient 直调（审阅所指的转场目标）
            try:
                o, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, attn_bias=None, compute_log_sumexp=True,
                    dropout_p=0.0, is_causal=True, scale=None)
                e = _err(o, ref)
                lse_slice = lse[..., :Sq] if lse.shape[-1] >= Sq else lse
                lse_err = (lse_slice.float() - ref_lse.float()).abs().max().item()
                rec(res, f"efficient/{tag}/{dtn(dt)}", _status(e, dt), e,
                    note=f"lse_shape={tuple(lse.shape)} lse_err={lse_err:.3e}",
                    lse_shape=list(lse.shape), lse_err=lse_err)
            except Exception as ex:
                rec(res, f"efficient/{tag}/{dtn(dt)}", "ERROR",
                    note=f"{type(ex).__name__}: {str(ex)[:110]}")


def check_constraints(res, dev):
    """_aligned_bias / GQA 扩 KV / 可微 mask 三条联动约束。"""
    from kernel.backends.thead_ppu import _aligned_bias

    # 4a) 非对齐 bias + 非方阵 causal（bool mask → additive → _aligned_bias）
    for tag, Sq, Skv, dt in (
        ("bool333x197", 333, 197, torch.float16),
        ("float100x197", 100, 197, torch.bfloat16),
        ("bool1x64", 1, 64, torch.float16),
    ):
        q, k, v, m = mk(1, 4, 4, Sq, Skv, 64, dt, dev, seed=7, mask="bool"
                        if tag.startswith("bool") else "float")
        bias = m
        if bias.dtype == torch.bool:
            bias = torch.zeros_like(bias, dtype=q.dtype)
            bias.masked_fill_(~m, float("-inf"))
        elif bias.dtype != q.dtype:
            bias = bias.to(q.dtype)
        aligned = _aligned_bias(bias)
        changed = aligned.shape != bias.shape or aligned.stride() != bias.stride()
        try:
            o, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, attn_bias=aligned, compute_log_sumexp=False,
                dropout_p=0.0, is_causal=True, scale=None)
            ref, _ = _ref(q, k, v, m, True, None)
            e = _err(o, ref)
            rec(res, f"aligned_bias_causal/{tag}", _status(e, dt), e,
                note=f"bias_repaired={bool(changed)} "
                     f"out_shape={tuple(o.shape)}")
        except Exception as ex:
            rec(res, f"aligned_bias_causal/{tag}", "ERROR",
                note=f"{type(ex).__name__}: {str(ex)[:110]}")

    # 4b) GQA 扩 KV + 非方阵 causal
    for tag, Hq, Hkv, Sq, Skv in (
        ("gqa8/2_1x64", 8, 2, 1, 64),
        ("gqa8/2_100x197", 8, 2, 100, 197),
    ):
        for dt in (torch.float16, torch.bfloat16):
            q, k, v, _ = mk(2, Hq, Hkv, Sq, Skv, 64, dt, dev, seed=11)
            rep = Hq // Hkv
            ke = k.repeat_interleave(rep, dim=1)
            ve = v.repeat_interleave(rep, dim=1)
            try:
                o, lse, *_ = \
                    torch.ops.aten._scaled_dot_product_efficient_attention(
                        q, ke, ve, attn_bias=None, compute_log_sumexp=True,
                        dropout_p=0.0, is_causal=True, scale=None)
                ref, _ = _ref(q, k, v, None, True, None)
                e = _err(o, ref)
                rec(res, f"gqa_efficient/{tag}/{dtn(dt)}", _status(e, dt),
                    e, note=f"lse_shape={tuple(lse.shape)}")
            except Exception as ex:
                rec(res, f"gqa_efficient/{tag}/{dtn(dt)}", "ERROR",
                    note=f"{type(ex).__name__}: {str(ex)[:110]}")


def check_autograd(res, dev):
    """厂商 efficient 反向 + A1 math backward 门控（register.py:109）。"""
    # 5a) 厂商 efficient 直调反向（非方阵 causal）
    q, k, v, _ = mk(1, 4, 4, 100, 197, 64, torch.float16, dev, seed=3)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    try:
        o, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, attn_bias=None, compute_log_sumexp=True,
            dropout_p=0.0, is_causal=True, scale=None)
        o.sum().backward()
        rq, rk, rv = (x.detach().requires_grad_(True) for x in (q, k, v))
        r, _ = _ref(rq, rk, rv, None, True, None)
        r.sum().backward()
        errs = [_err(g, rg) for g, rg in
                ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad))]
        rec(res, "efficient_backward/100x197-fp16",
            "PASS" if max(errs) < TOL["float16"] else "FAIL", max(errs),
            note=f"dq={errs[0]:.3e} dk={errs[1]:.3e} dv={errs[2]:.3e}")
    except Exception as ex:
        rec(res, "efficient_backward/100x197-fp16", "ERROR",
            note=f"{type(ex).__name__}: {str(ex)[:140]}")

    # 5b) A1 math backward 旧门回放: 整改前 register.py:109 是
    #     `if ctx.is_causal and Sq == Skv` —— 下面**内联复刻**旧门体
    #     （不是现行代码），故修后重跑仍报 MISMATCH，用来保留「为什么要
    #     拆门」的证据；现行路径的梯度判卷见 test/op_level.py 第 6 步。
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sdpa_register_probe", OP_DIR / "register.py")
        reg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reg)
        q2, k2, v2, _ = mk(1, 4, 4, 100, 197, 64, torch.float32, dev, seed=3)
        q2 = q2.detach().requires_grad_(True)
        k2 = k2.detach().requires_grad_(True)
        v2 = v2.detach().requires_grad_(True)

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.is_causal, ctx.attn_mask, ctx.scale = True, None, None
        r2, _ = _ref(q2, k2, v2, None, True, None)
        r2.sum().backward()
        # 复算**整改前** register.py backward 正文（含 Sq==Skv 旧门）
        B_, Hq, Sq, D = q2.shape
        _, Hkv, Skv, _ = k2.shape
        gqa_rep = Hq // Hkv
        qf, kf, vf = q2.float(), k2.float(), v2.float()
        do = torch.ones_like(r2)
        s = D ** -0.5
        if gqa_rep != 1:
            kf = kf.repeat_interleave(gqa_rep, dim=1)
            vf = vf.repeat_interleave(gqa_rep, dim=1)
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * s
        if ctx.is_causal and Sq == Skv:
            causal = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool,
                                           device=qf.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        dv = torch.matmul(p.transpose(-1, -2), do)
        dp = torch.matmul(do, vf.transpose(-1, -2))
        da = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
        dq = torch.matmul(da, kf) * s
        dk = torch.matmul(da.transpose(-1, -2), qf) * s
        errs = [_err(dq, q2.grad), _err(dk, k2.grad), _err(dv, v2.grad)]
        rec(res, "a1_math_backward_gate/100x197", "MISMATCH"
            if max(errs) > TOL["float32"] else "PASS", max(errs),
            note=f"dq={errs[0]:.3e} dk={errs[1]:.3e} dv={errs[2]:.3e} "
                 f"（旧门回放: 整改前 register.py:109 跳过非方阵 causal，"
                 f"现门已拆，现行判卷见 test/op_level.py 第 6 步）")
    except Exception as ex:
        rec(res, "a1_math_backward_gate/100x197", "ERROR",
            note=f"{type(ex).__name__}: {str(ex)[:140]}")
        traceback.print_exc()


def check_reference_gate(res, dev):
    """判卷标准侧 reference 是否接受非方阵 causal（整改前是 assert 拒绝,
    见 prefix 快照的 reference_nonsq_causal/REJECT 行）。"""
    from reference import sdpa_reference
    q, k, v, _ = mk(1, 4, 4, 4, 8, 64, torch.float32, "cpu", seed=5)
    try:
        sdpa_reference(q, k, v, None, 0.0, True, None, False)
        rec(res, "reference_nonsq_causal", "ACCEPTED")
    except Exception as e:
        rec(res, "reference_nonsq_causal", "REJECT",
            note=f"{type(e).__name__}: {str(e)[:110]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:6")
    args = ap.parse_args()
    dev = args.device
    res: dict = {"device": dev, "checks": [], "misc": {}}
    checks = res["checks"]

    res["misc"]["device_name"] = torch.cuda.get_device_name(
        torch.device(dev))
    res["misc"]["torch"] = torch.__version__

    check_reference_gate(checks, dev)
    check_shapes(checks, dev)
    check_constraints(checks, dev)
    check_autograd(checks, dev)

    tally: dict = {}
    for c in checks:
        tally[c["status"]] = tally.get(c["status"], 0) + 1
    res["summary"] = tally
    print("\nSUMMARY", json.dumps(tally, ensure_ascii=False))

    out_dir = OP_DIR / "probes" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thead_nonsq_causal_probe.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out_dir / "thead_nonsq_causal_probe.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
