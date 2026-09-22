# 应用层验证（轻量消费方）: mini-decoder 前向双跑。
#
# 本栈 vllm 为空壳，按 FlagOS-OP softmax-fullstack 的先例（"attention
# scores 消费 softmax"）构建 Python 层真实消费方:
#   - MiniDecoder: Llama 风格 4 层（GQA 8/2 头 · causal · LayerNorm · MLP）
#   - 业务代码只调 F.scaled_dot_product_attention（标准 aten 路径，零改动）
# 双跑断言:
#   ①拦截计数 > 0（组合计算链路真的用上了自研实现）
#   ②logits 数值一致（两实现数学等价，低精度容差）
#   ③贪心 next-token 行为一致（top-1 全位置对齐 + 续写序列一致率，
#     阈值按 FlagOS-OP 断言策略: 数值微差在深层可被贪心解码混沌放大）
#
# 运行: python3 test/framework_level.py [ascend910|p800-kunlunxin]
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

_CALLS = {"n": 0}


def _build_model(dev, dtype):
    import torch
    import torch.nn as nn

    class MiniAttention(nn.Module):
        def __init__(self, d, nh, nkv):
            super().__init__()
            self.nh, self.nkv, self.hd = nh, nkv, d // nh
            self.wq = nn.Linear(d, nh * self.hd, bias=False)
            self.wk = nn.Linear(d, nkv * self.hd, bias=False)
            self.wv = nn.Linear(d, nkv * self.hd, bias=False)
            self.wo = nn.Linear(nh * self.hd, d, bias=False)

        def forward(self, x):
            B, S, _ = x.shape
            q = self.wq(x).view(B, S, self.nh, self.hd).transpose(1, 2)
            k = self.wk(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
            v = self.wv(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
            # ← 应用层核心: 标准 aten 调用（业务代码零改动）
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True)
            return self.wo(o.transpose(1, 2).reshape(B, S, -1))

    class Block(nn.Module):
        def __init__(self, d, nh, nkv, hidden):
            super().__init__()
            self.attn = MiniAttention(d, nh, nkv)
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class MiniDecoder(nn.Module):
        def __init__(self, vocab=512, d=256, layers=4, nh=8, nkv=2,
                     hidden=1024, max_s=128):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(max_s, d)
            self.blocks = nn.ModuleList(
                Block(d, nh, nkv, hidden) for _ in range(layers))
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, ids):
            S = ids.shape[1]
            h = self.tok(ids) + self.pos.weight[:S]
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln(h))

    torch.manual_seed(20260917)
    m = MiniDecoder()
    return m.to(dev).to(dtype).eval()


def _greedy(model, ids, steps):
    import torch
    with torch.no_grad():
        cur = ids.clone()
        for _ in range(steps):
            nxt = model(cur)[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    return cur[:, ids.shape[1]:]           # 续写部分


def run(profile):
    import torch
    if profile.torch_device.startswith("npu"):
        import torch_npu  # noqa: F401

    from register import register_a1

    dev = profile.torch_device
    # P800 的 nn.Linear/LayerNorm fp16 链路在本随机 mini-decoder 上溢出；
    # 应用语义仍是低精度注意力，使用栈上稳定的 bf16。
    dt = torch.bfloat16 if profile.vendor == "kunlunxin" else torch.float16
    N_PROMPTS, S0, STEPS = 4, 48, 6

    g = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, 512, (N_PROMPTS, S0), generator=g).to(dev)

    # ── 基线跑（未注册 → 原生 SDPA）──
    model = _build_model(dev, dt)
    with torch.no_grad():
        logits_base = model(ids)
    cont_base = _greedy(model, ids, STEPS)
    top1_base = logits_base.argmax(dim=-1)

    # ── 插件跑（A1 注册 → 自研 Triton 接管 aten 路径）──
    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        logits_plug = model(ids)
    cont_plug = _greedy(model, ids, STEPS)

    # ① 拦截命中: 每 forward 4 层 × 1 次 = 4；7 次 forward（1+贪心 6）
    assert _CALLS["n"] >= 4, f"消费链路未用上自研实现: {_CALLS['n']}"

    # ② logits 数值一致（低精度容差，依据实测误差见 accuracy 分册）
    diff = (logits_plug.float() - logits_base.float()).abs().max().item()
    tol = 5e-2
    assert diff < tol, f"logits 数值分歧超容差: {diff}"

    # ③ 行为一致: 全位置 top-1 对齐 + 贪心续写序列一致率
    top1_plug = logits_plug.argmax(dim=-1)
    top1_rate = (top1_plug == top1_base).float().mean().item()
    assert top1_rate == 1.0, f"top-1 一致率 {top1_rate}"

    seq_match = (cont_plug == cont_base).all(dim=1)   # 每 prompt 整串一致
    seq_rate = seq_match.float().mean().item()
    # FlagOS-OP 断言策略: 自定义数值实现允许混沌分叉，一致率 ≥ 2/3
    assert seq_rate >= 2 / 3, f"贪心续写一致率 {seq_rate} < 2/3"

    print(f"  消费方: mini-decoder 4层 GQA8/2 causal {str(dt).removeprefix('torch.')} "
          f"({N_PROMPTS} prompts × {S0}+{STEPS} tokens)")
    print(f"  拦截: count={_CALLS['n']} (≥4 层调用)")
    print(f"  logits max_diff={diff:.3e} (<{tol}) · top-1 一致率 "
          f"{top1_rate:.2f} · 续写一致率 {seq_rate:.2f} (≥2/3)")
    return {"ok": True, "intercepted": _CALLS["n"],
            "logits_max_diff": round(diff, 6),
            "top1_rate": top1_rate, "greedy_seq_match": seq_rate,
            "consumer": "mini-decoder(4L GQA8/2 causal)"}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else None)))
