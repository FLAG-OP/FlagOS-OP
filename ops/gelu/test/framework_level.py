# 应用层验证: mini Transformer FFN consumer（gelu 消费位）双跑。
#
# 口径（issue #55 交付要求 3 + docs/testing.md framework 层）:
#   - 同 prompt/同 seed，唯一变量是 A1 注入 → 输出差异可归因到 gelu;
#   - 自研 Triton gelu 在 fp32 参考容差内不改数值语义 → 恒等断言:
#     注入前后 logits/续写序列/梯度逐位一致（op 层已对拍解析反向）;
#   - 调用计数 > 0 且随续写步数增长（真实消费证据, pid 分片双通道）;
#   - thead-ppu dev 栈结构性无 flag_gems（thead-ppu.lock.yaml notes，
#     T1 sdpa 同款缺证口径）→ 仅拦截/数值一致性，跳过 FlagOS 栈门禁。
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

    class GeluFFN(nn.Module):
        """Transformer FFN: Linear(256→1024) → GELU → Linear(1024→256)。

        输出维度 = 输入维度，支持特征自回归续写（每步真实消费 gelu，
        调用计数随之增长——framework 层"被消费且不破坏输出"的动态证据）。
        """

        def __init__(self, d=256):
            super().__init__()
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    torch.manual_seed(20261125)
    return GeluFFN().to(dev).to(dtype).eval()


def _autoregressive(model, x, steps):
    """特征自回归: 每步把末位输出追加为新位置（每步消费一次 gelu）。"""
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
    # bf16 避免 fp16 随机 Linear 链路溢出（embedding framework 同款口径）。
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = torch.randn(4, 48, 256, generator=generator).to(dev).to(dtype)

    model = _build_model(dev, dtype)

    # ── baseline（注册前）: forward / 自回归续写 / backward 梯度 ──
    with torch.no_grad():
        base_logits = model(x)
    base_cont = _autoregressive(model, x, 6)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    base_g1 = model.fc1.weight.grad.detach().clone()
    base_g2 = model.fc2.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    # ── injected: A1 注册 gelu 后同输入双跑 ──
    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        inj_logits = model(x)
    inj_cont = _autoregressive(model, x, 6)

    logits_diff = (inj_logits.float() - base_logits.float()).abs().max().item()
    assert logits_diff == 0.0, f"logits diff={logits_diff}"
    top1 = torch.equal(inj_logits.argmax(-1), base_logits.argmax(-1))
    assert top1, "top1 不一致"
    cont_match = torch.equal(inj_cont, base_cont)
    assert cont_match, "自回归续写不一致（混沌分叉应归因实现, 恒等实现不应分叉）"
    assert _CALLS["n"] >= 7, \
        f"A1 计数 {_CALLS['n']} < 7（1 前向 + 6 续写步, gelu 未被真实消费）"

    # ── backward 一致性: 注入后训练一步梯度 vs baseline ──
    loss = model(x).float().pow(2).mean()
    loss.backward()
    g1_diff = (model.fc1.weight.grad.float() - base_g1.float()).abs().max().item()
    g2_diff = (model.fc2.weight.grad.float() - base_g2.float()).abs().max().item()
    assert g1_diff < 1e-2, f"fc1 grad diff={g1_diff}"
    assert g2_diff < 1e-2, f"fc2 grad diff={g2_diff}"

    return {"ok": True,
            "consumer": "mini FFN (Linear+GELU+Linear, bf16, 6-step autoreg)",
            "logits_diff": logits_diff,
            "top1_match": top1,
            "continuation_match": cont_match,
            "grad_max_diff": max(g1_diff, g2_diff),
            "gelu_calls": _CALLS["n"]}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
