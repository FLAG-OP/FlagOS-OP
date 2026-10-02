"""端到端负载: MiniDecoder（4 层 × D512 × 8 头 + 熵正则 P 消费者）。

场景 prefill : S=1024 全前向, 每层注意力消费 P（蒸馏式负载）
场景 decode  : 重算式贪心 128 步（ctx 512→639, 同仓库应用层测试口径）, 每层消费 P
场景 cached  : KV-cache 预填 1024 + 128 步逐 token 解码（真实服务口径, 步内消费 P）

模式（各跑独立子进程, torch.library 注册不可撤销）:
  ours   : register_a1 后直调 aten::_scaled_dot_product_attention_math → 自研 Triton
  native : 不注册, 同一调用 → 原生 math 后端（同契约, 都返回 P）
  fsdpa  : F.sdpa 默认融合后端 —— 不产出 P, 契约不同, 仅作天花板参照
  noop   : 注意力输出置零（纯投影+MLP+头基线, 用于算注意力占比）
"""
import json
import os
import statistics
import sys
import time

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
import torch_npu

DEV, DT = "npu:0", torch.float16
N_LAYERS, D, H, HEAD, V = 4, 512, 8, 64, 512
PREFILL_S, CTX0, STEPS = 1024, 512, 128


def do_attn(q, k, v, mode, causal):
    if mode == "noop":
        return torch.zeros_like(q), None
    if mode == "fsdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal), None
    return torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, causal, None)


def entropy(p):
    return -(p * p.clamp_min(1e-9).log()).sum(-1).mean()


class Mini:
    def __init__(self, dev, seed=0):
        g = torch.Generator("cpu").manual_seed(seed)
        self.emb = (torch.randn(V, D, generator=g) * 0.1).to(dev).to(DT)
        mk = lambda a, b: (torch.randn(a, b, generator=g) * 0.05).to(dev).to(DT)
        self.layers = [
            dict(Wq=mk(D, D), Wk=mk(D, D), Wv=mk(D, D), Wo=mk(D, D),
                 W1=mk(D, 4 * D), W2=mk(4 * D, D))
            for _ in range(N_LAYERS)]
        self.Wout = mk(D, V)

    # ---- 场景 1/2: 全量重算 ----
    def forward(self, tokens, mode, consume_p=True):
        x = self.emb[tokens]
        for L in self.layers:
            B, S, _ = x.shape
            q = (x @ L["Wq"]).view(B, S, H, HEAD).transpose(1, 2)
            k = (x @ L["Wk"]).view(B, S, H, HEAD).transpose(1, 2)
            v = (x @ L["Wv"]).view(B, S, H, HEAD).transpose(1, 2)
            ctx, p = do_attn(q, k, v, mode, causal=True)
            if consume_p and p is not None:
                entropy(p)
            x = x + (ctx.transpose(1, 2).reshape(B, S, D) @ L["Wo"])
            x = x + F.gelu(x @ L["W1"]) @ L["W2"]
        return (x @ self.Wout) @ self.emb.t()

    # ---- 场景 3: KV cache ----
    def prefill_cache(self, tokens, mode):
        x = self.emb[tokens]                                # (1,CTX0,D)
        caches = []
        S = x.shape[1]
        for L in self.layers:
            k = (x @ L["Wk"]).view(1, S, H, HEAD).transpose(1, 2)
            v = (x @ L["Wv"]).view(1, S, H, HEAD).transpose(1, 2)
            caches.append([k, v])
            q = (x @ L["Wq"]).view(1, S, H, HEAD).transpose(1, 2)
            ctx, p = do_attn(q, k, v, mode, causal=True)
            if p is not None:
                entropy(p)
            x = x + (ctx.transpose(1, 2).reshape(1, S, D) @ L["Wo"])
            x = x + F.gelu(x @ L["W1"]) @ L["W2"]
        return (x @ self.Wout) @ self.emb.t(), caches

    def decode_cached(self, x1, caches, mode):
        for L, (k, v) in zip(self.layers, caches):
            q = (x1 @ L["Wq"]).view(1, 1, H, HEAD).transpose(1, 2)
            kt = (x1 @ L["Wk"]).view(1, 1, H, HEAD).transpose(1, 2)
            vt = (x1 @ L["Wv"]).view(1, 1, H, HEAD).transpose(1, 2)
            k = torch.cat([k, kt], dim=2)
            v = torch.cat([v, vt], dim=2)
            caches_cache = (k, v)
            ctx, p = do_attn(q, k, v, mode, causal=False)
            if p is not None:
                entropy(p)
            x1 = x1 + (ctx.transpose(1, 2).reshape(1, 1, D) @ L["Wo"])
            x1 = x1 + F.gelu(x1 @ L["W1"]) @ L["W2"]
        return (x1 @ self.Wout) @ self.emb.t(), caches


def bench(fn, warm, iters):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


ONLY = os.environ.get("ONLY","")


def main(mode):
    global _LIBS, CTR
    CTR = None
    if mode == "ours":
        from register import register_a1
        CTR = {"n": 0}
        _LIBS = register_a1("AutogradPrivateUse1", CTR, "triton")
    m = Mini(DEV)
    with torch.no_grad():
        ptok = torch.zeros(1, PREFILL_S, dtype=torch.long, device=DEV)
        m.forward(ptok, mode, consume_p=(mode not in ("fsdpa", "noop")))
        t_pre = 0
        if not ONLY or ONLY == "re":
            t_pre = bench(lambda: m.forward(
                ptok, mode, consume_p=(mode not in ("fsdpa", "noop"))), 3, 10)

        def greedy():
            seq = [torch.zeros(CTX0, dtype=torch.long, device=DEV)]
            for _ in range(STEPS):
                toks = torch.cat(seq, dim=0).unsqueeze(0)
                logits = m.forward(toks, mode,
                                   consume_p=(mode not in ("fsdpa", "noop")))
                nxt = int(logits[0, -1].argmax().item())
                seq.append(torch.tensor([nxt], device=DEV))
            return len(seq)

        t_dec = 0
        if not ONLY or ONLY == "re":
            greedy()
            t_dec = bench(greedy, 1, 3)

        def cached():
            logits, caches = m.prefill_cache(ptok, mode)
            tok = torch.tensor([[int(logits[0, -1].argmax().item())]],
                               device=DEV)
            x1 = m.emb[tok]
            for _ in range(STEPS):
                logits, caches = m.decode_cached(
                    x1, caches, mode)
                nxt = int(logits[0, -1].argmax().item())
                tok = torch.tensor([[nxt]], device=DEV)
                x1 = m.emb[tok]
            return len(caches)

        t_cch = 0
        if not ONLY or ONLY == "cached":
            cached()
            t_cch = bench(cached, 1, 3)
    print(json.dumps({"mode": mode, "hits": (CTR or {}).get("n"),
                      "prefill_ms": round(t_pre, 1),
                      "decode_re_ms": round(t_dec, 1),
                      "decode_cache_ms": round(t_cch, 1)}))


if __name__ == "__main__":
    main(sys.argv[1])
