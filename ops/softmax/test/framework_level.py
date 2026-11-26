# 应用层验证: attention-scores consumer（softmax 消费位）双跑。
#
# consumer 选 mini-attention: QK^T 后沿 S 维 softmax——softmax 在真实
# 推理链路的典型消费位（issue #55 台账: 1,598 次调用 / 109 模型）。
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

    class MiniAttention(nn.Module):
        """单头 mini-attention: softmax(QK^T/√d)·V（softmax 消费位）。"""

        def __init__(self, d=128):
            super().__init__()
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.proj = nn.Linear(d, d)
            self.d = d

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = q @ k.transpose(-1, -2) / (self.d ** 0.5)
            attn = torch.nn.functional.softmax(scores, dim=-1)
            return self.proj(attn @ v)

    torch.manual_seed(20261125)
    return MiniAttention().to(dev).to(dtype).eval()


def _autoregressive(model, x, steps):
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
    x = torch.randn(2, 64, 128, generator=generator).to(dev).to(dtype)

    model = _build_model(dev, dtype)

    with torch.no_grad():
        base_out = model(x)
    base_cont = _autoregressive(model, x, 4)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    base_gq = model.q.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        inj_out = model(x)
    inj_cont = _autoregressive(model, x, 4)

    out_diff = (inj_out.float() - base_out.float()).abs().max().item()
    assert out_diff == 0.0, f"attention 输出 diff={out_diff}"
    assert torch.equal(inj_cont, base_cont), "自回归续写不一致"
    assert _CALLS["n"] >= 5, \
        f"A1 计数 {_CALLS['n']} < 5（softmax 未被真实消费）"

    loss = model(x).float().pow(2).mean()
    loss.backward()
    g_diff = (model.q.weight.grad.float() - base_gq.float()).abs().max().item()
    assert g_diff < 1e-2, f"grad diff={g_diff}"

    return {"ok": True,
            "consumer": "mini-attention softmax(scores) (bf16)",
            "out_diff": out_diff,
            "continuation_match": True,
            "grad_max_diff": g_diff,
            "softmax_calls": _CALLS["n"]}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
