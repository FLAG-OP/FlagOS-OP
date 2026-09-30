# framework 层: 让"真实消费方"跑在本算子上——两套迷你消费者双跑
# （注册前原生基线 vs 注册后 A1 路径），对 logits / 生成序列 / 概率图
# 消费指标 / 梯度做回归比对，并统计拦截次数。
#
# 运行: python3 test/framework_level.py [ascend910|cpu]
#
# 消费方 A — MiniDecoder: 两层 attention-block + tied embedding 的小解码器，
#   注意力直调 `torch.ops.aten._scaled_dot_product_attention_math`
#   （NPU 上 F.sdpa 被 torch_npu 路由到融合注意力、从不进本算子，
#   直调是 NPU 上唯一真实消费方；CPU 上同一行同样命中注册）。
# 消费方 B — probs_head: 消费**第二输出概率图**（熵正则），反向传播到
#   全部权重——验证双输出算子的 dprobs 通路在真实图里可用。
from __future__ import annotations

import sys
from pathlib import Path

import torch

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

VOCAB, D, H, HEAD = 32, 64, 4, 16


def _attn(x, Wq, Wk, Wv, causal=True, gqa=False):
    """单头组注意力块: x(B,S,D) → 上下文(B,S,D)。"""
    B, S, _ = x.shape
    Hkv = H // 2 if gqa else H
    q = (x @ Wq).view(B, S, H, HEAD).transpose(1, 2)
    k = (x @ Wk).view(B, S, Hkv, HEAD).transpose(1, 2)
    v = (x @ Wv).view(B, S, Hkv, HEAD).transpose(1, 2)
    ctx, _ = torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, causal, None, enable_gqa=gqa)
    return ctx.transpose(1, 2).reshape(B, S, D)


def _decoder(emb, tokens, weights, use_probs=False):
    """两块 MiniDecoder。返回 (logits, probs|None)。"""
    Wq1, Wk1, Wv1, Wo1, Wq2, Wk2, Wv2, Wo2, Wout = weights
    x = emb[tokens]                      # (B,S,D)
    x = x + (_attn(x, Wq1, Wk1, Wv1) @ Wo1)
    x = x + (_attn(x, Wq2, Wk2, Wv2) @ Wo2)
    logits = (x @ Wout) @ emb.t()        # 输出投影 + tied embedding 头
    probs = None
    if use_probs:
        # 概率图消费者: 第二输出 (B,H,S,S) 上的熵正则 + logits CE
        q = (x @ Wq1).view(x.shape[0], x.shape[1], H, HEAD).transpose(1, 2)
        k = (x @ Wk1).view(x.shape[0], x.shape[1], H, HEAD).transpose(1, 2)
        v = (x @ Wv1).view(x.shape[0], x.shape[1], H, HEAD).transpose(1, 2)
        _, probs = torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, True, None, enable_gqa=False)
    return logits, probs


def _mk_weights(dev, seed=0):
    g = torch.Generator("cpu").manual_seed(seed)
    emb = (torch.randn(VOCAB, D, generator=g) * 0.1).to(dev)
    ws = tuple((torch.randn(D, D, generator=g) * 0.08).to(dev)
               for _ in range(9))
    return emb, ws


def _greedy(emb, ws, dev, T=8, steps=8):
    """逐 token 贪心解码（无 grad），返回生成序列 + 各步 logits。"""
    toks = [torch.zeros(1, dtype=torch.long, device=dev)]  # BOS=0
    with torch.no_grad():
        for _ in range(steps):
            seq = torch.cat(toks, dim=0).unsqueeze(0)      # (1,S)
            logits, _ = _decoder(emb, seq, ws)
            nxt = int(logits[0, -1].argmax().item())
            toks.append(torch.tensor([nxt], device=dev))
    return [int(t.item()) for t in toks[1:]]


def run(profile):
    import torch

    from register import register_a1

    dev = profile.torch_device
    results = []

    def check(name, cond, extra=""):
        mark = "OK  " if cond else "FAIL"
        results.append(f"  [{mark}] {name}" + (f" — {extra}" if extra else ""))
        assert cond, f"{name}: {extra}"

    # 固定输入（CPU 生成 → .to(dev)，两次跑完全同一批张量）
    g = torch.Generator("cpu").manual_seed(2026)
    tokens = torch.randint(0, VOCAB, (1, 12), generator=g).to(dev)
    grads = torch.Generator("cpu").manual_seed(7)
    dw = tuple((torch.randn(D, D, generator=grads) * 0.05).to(dev)
               for _ in range(9))
    emb, ws = _mk_weights(dev)
    # 两套独立参数图（基线 vs 接管后），权重数值相同、梯度互不干扰
    emb_b, ws_b = emb.clone(), tuple(w.clone() for w in ws)
    emb = emb.requires_grad_(True)
    emb_b = emb_b.requires_grad_(True)
    requires = tuple(w.requires_grad_(True) for w in ws)
    ws_b = tuple(w.requires_grad_(True) for w in ws_b)

    # ── 0) 注册前: 原生基线（logits / probs-熵 / 梯度 / 贪心序列）──
    logits_b, probs_b = _decoder(emb_b, tokens, ws_b, use_probs=True)
    # clamp: causal 下三角外 P=0，log(0)=-inf → 0·(-inf)=nan
    loss_b = (logits_b.square().mean()
              + 0.1 * (probs_b * probs_b.clamp_min(1e-9).log()).mean())
    grads_b = torch.autograd.grad(loss_b, (emb_b, *ws_b), retain_graph=False)
    seq_b = _greedy(emb_b.detach(), ws_b, dev)

    # ── 1) 注册 A1（保留 lib 引用）──
    ctr: dict = {"n": 0}
    libs = register_a1(profile.dispatch_key, ctr, profile.default_impl)

    # ── 2) 消费方 A/B 重跑（同种子同权重同 token）──
    logits_h, probs_h = _decoder(emb, tokens, ws, use_probs=True)
    loss_h = (logits_h.square().mean()
              + 0.1 * (probs_h * probs_h.clamp_min(1e-9).log()).mean())
    grads_h = torch.autograd.grad(loss_h, (emb, *requires))

    n_after_fwd = ctr["n"]
    check("消费者 A+B 前向命中本算子", n_after_fwd > 0, f"intercept={n_after_fwd}")

    d_log = (logits_h - logits_b.detach()).abs().max().item()
    check("logits 与原生基线一致", d_log < 1e-4, f"L∞={d_log:.2e}")

    top1 = (logits_h.argmax(-1) == logits_b.detach().argmax(-1)).float().mean()
    check("top-1 命中率 ≥ 0.95", float(top1) >= 0.95,
          f"top1={float(top1):.3f}")

    d_prob = (probs_h - probs_b.detach()).abs().max().item()
    check("概率图与原生基线一致", d_prob < 1e-5, f"P L∞={d_prob:.2e}")

    d_g = max((a - b.detach()).abs().max().item()
              for a, b in zip(grads_h, grads_b))
    check("全权重梯度（含 dprobs 通路）= 原生基线", d_g < 1e-4,
          f"∇ L∞={d_g:.2e}")

    # ── 3) 贪心解码回归（注册后重新生成，逐步比对）──
    seq_h = _greedy(emb.detach(), ws, dev)
    hit = sum(a == b for a, b in zip(seq_h, seq_b)) / len(seq_b)
    check("贪心序列逐步命中 ≥ 2/3", hit >= 2 / 3,
          f"{hit:.3f} seq={seq_h} vs {seq_b}")

    # ── 4) inference 消费方（无 grad 推理路径也应拦截）──
    n0 = ctr["n"]
    with torch.inference_mode():
        _decoder(emb.detach(), tokens, ws)
    check("inference_mode 消费方命中", ctr["n"] > n0, f"+{ctr['n'] - n0}")

    check("lib 引用保留（注册未被 GC）", libs and len(libs) > 0)

    for ln in results:
        print(ln)
    return {"ok": True, "dispatch_key": profile.dispatch_key,
            "impl": profile.default_impl, "intercepts": ctr["n"],
            "top1": round(float(top1), 4), "greedy_hit": round(hit, 4),
            "logit_linf": round(d_log, 9), "grad_linf": round(d_g, 9),
            "cases": len(results)}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
