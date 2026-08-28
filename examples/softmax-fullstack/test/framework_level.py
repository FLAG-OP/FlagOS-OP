# 应用层验证: attention scores 消费者真实触发 softmax。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def run(profile):
    import torch
    import torch.nn.functional as F

    from register import CALLS

    dev = profile.torch_device
    torch.manual_seed(42)
    B, H, T, D = 1, 4, 64, 64
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=dev)
    k = q.clone()

    def attn_scores(Q, K):
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / (D ** 0.5)
        return F.softmax(scores, dim=-1)

    with torch.no_grad():
        ref = attn_scores(q, k)
        out = attn_scores(q, k)
        err = (out.float() - ref.float()).abs().max().item()
    n = CALLS["softmax"]
    assert n >= 2, f"attention scores 未触发 softmax ({n})"
    assert err < 1e-2, f"应用层偏差 {err}"
    return {"ok": True, "calls": n, "max_err": round(err, 8)}
