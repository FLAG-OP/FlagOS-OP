# 应用层验证: mini Transformer block consumer（LayerNorm 消费位）双跑。
#
# issue #55 交付要求 5: 两个 LayerNorm 路径都要分别证明真实消费——
# 本 consumer 同时含 F.layer_norm（公开路径, 内部消费 native_layer_norm）
# 与残差结构, 与 ops/layer_norm 的 framework 消费者互补。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

_CALLS = {"n": 0}


def _build_model(dev, dtype):
    import torch
    import torch.nn as nn

    class LNormBlock(nn.Module):
        """Pre-LN Transformer block 骨架: LN → Linear → GELU → Linear + 残差。"""

        def __init__(self, d=256):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, 512)
            self.fc2 = nn.Linear(512, d)

        def forward(self, x):
            return x + self.fc2(
                torch.nn.functional.gelu(self.fc1(self.ln(x))))

    torch.manual_seed(20261125)
    return LNormBlock().to(dev).to(dtype).eval()


def _autoregressive(model, x, steps):
    """特征自回归: 每步把末位输出追加为新位置（每步真实消费 LN）。"""
    import torch

    with torch.no_grad():
        cur = x.clone()
        for _ in range(steps):
            nxt = model(cur)[:, -1:, :]
            cur = torch.cat([cur, nxt], dim=1)
    return cur


def run(profile):
    import torch

    from register import register_a1

    dev = profile.torch_device
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = torch.randn(4, 48, 256, generator=generator).to(dev).to(dtype)

    model = _build_model(dev, dtype)

    with torch.no_grad():
        base_logits = model(x)
    base_cont = _autoregressive(model, x, 6)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    base_g = model.ln.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        inj_logits = model(x)
    inj_cont = _autoregressive(model, x, 6)

    logits_diff = (inj_logits.float() - base_logits.float()).abs().max().item()
    assert logits_diff == 0.0, f"logits diff={logits_diff}"
    assert torch.equal(inj_logits.argmax(-1), base_logits.argmax(-1)), "top1"
    assert torch.equal(inj_cont, base_cont), "自回归续写不一致"
    assert _CALLS["n"] > 0, "A1 计数为 0（native_layer_norm 未被消费）"

    loss = model(x).float().pow(2).mean()
    loss.backward()
    g_diff = (model.ln.weight.grad.float() - base_g.float()).abs().max().item()
    assert g_diff < 1e-2, f"grad diff={g_diff}"

    return {"ok": True,
            "consumer": "pre-LN block (LN+Linear+GELU+Linear, bf16)",
            "logits_diff": logits_diff,
            "greedy_match": True,
            "grad_max_diff": g_diff,
            "native_layer_norm_calls": _CALLS["n"]}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
