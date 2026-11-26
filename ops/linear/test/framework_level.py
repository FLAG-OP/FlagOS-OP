# 应用层验证: mini Transformer FFN consumer（Linear 消费位）双跑。
#
# 口径（issue #55 交付要求 5: linear 需单独证明真实消费）:
#   - 同 prompt/同 seed, 唯一变量是 A1 注入 → 输出差异可归因到 linear;
#   - 委托原生 → 数值恒等, logits 逐位一致;
#   - 调用计数 > 0（counter + pid 分片文件双通道）;
#   - thead-ppu dev 栈无 flag_gems（thead-ppu.lock.yaml notes）→
#     仅拦截/数值一致性, 跳过 FlagOS 栈门禁（T1 sdpa 同款缺证口径）。
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

    class FFNBlock(nn.Module):
        """Transformer FFN: Linear → GELU(native) → Linear（d 入=出,
        支持特征自回归续写）。"""

        def __init__(self, d=256):
            super().__init__()
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    torch.manual_seed(20261125)
    return FFNBlock().to(dev).to(dtype).eval()


def _autoregressive(model, x, steps):
    """特征自回归: 每步把末位输出追加为新位置（每步真实消费 linear）。"""
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

    # ── baseline（注册前）──
    with torch.no_grad():
        base_logits = model(x)
    base_cont = _autoregressive(model, x, 6)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    base_g = model.fc1.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    # ── injected: A1 注册 linear 后同输入双跑 ──
    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        inj_logits = model(x)
    inj_cont = _autoregressive(model, x, 6)

    logits_diff = (inj_logits.float() - base_logits.float()).abs().max().item()
    assert logits_diff == 0.0, f"logits diff={logits_diff}"
    assert torch.equal(inj_logits.argmax(-1), base_logits.argmax(-1)), "top1"
    assert torch.equal(inj_cont, base_cont), "自回归续写不一致"
    assert _CALLS["n"] > 0, "A1 计数为 0（linear 未被消费）"

    # backward 一致性（原生 autograd 节点随委托自动建立）
    loss = model(x).float().pow(2).mean()
    loss.backward()
    g_diff = (model.fc1.weight.grad.float() - base_g.float()).abs().max().item()
    assert g_diff == 0.0, f"grad diff={g_diff}"

    return {"ok": True,
            "consumer": "mini FFN (Linear+GELU+Linear, bf16)",
            "logits_diff": logits_diff,
            "top1_match": True,
            "greedy_match": True,
            "grad_max_diff": g_diff,
            "linear_calls": _CALLS["n"]}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
