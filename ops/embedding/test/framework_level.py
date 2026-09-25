#!/usr/bin/env python3
"""Application-level consumer: an embedding table plus token classifier."""
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

    class TokenMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=512, embedding_dim=256
            )
            self.mlp = nn.Sequential(
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 128),
            )

        def forward(self, ids):
            return self.mlp(self.embedding(ids))

    torch.manual_seed(20260923)
    return TokenMLP().to(dev).to(dtype).eval()


def _greedy(model, ids, steps):
    import torch

    with torch.no_grad():
        cur = ids.clone()
        for _ in range(steps):
            nxt = model(cur)[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    return cur[:, ids.shape[1]:]


def run(profile):
    import torch

    from register import register_a1

    dev = profile.torch_device
    # Framework/application validation is deliberately run inside the FlagOS
    # operator stack.  Enable the stable FlagGems GELU used by this consumer;
    # enabling the full op set is not deterministic on the locked P800 image,
    # and the same narrow FlagGems path keeps both platforms comparable.
    import flag_gems
    flag_gems.only_enable(include=["gelu"])
    assert flag_gems.current_work_registrar is not None
    assert "gelu" in flag_gems.current_work_registrar.include_ops

    # bf16 avoids overflow in the random fp16 Linear/GELU consumer.
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(19)
    ids = torch.randint(
        0, 512, (4, 48), generator=generator
    ).to(dev)

    model = _build_model(dev, dtype)

    # Capture native forward/backward before process-wide A1 registration.
    with torch.no_grad():
        baseline_logits = model(ids)
    baseline_cont = _greedy(model, ids, 6)
    loss = model(ids).float().pow(2).mean()
    loss.backward()
    baseline_embedding_grad = model.embedding.weight.grad.detach().clone()
    baseline_head_grad = model.mlp[0].weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        plugin_logits = model(ids)
    plugin_cont = _greedy(model, ids, 6)

    logits_diff = (
        plugin_logits.float() - baseline_logits.float()
    ).abs().max().item()
    assert logits_diff == 0.0, logits_diff
    top1_match = torch.equal(
        plugin_logits.argmax(dim=-1), baseline_logits.argmax(dim=-1)
    )
    assert top1_match
    seq_match = (plugin_cont == baseline_cont).all(dim=1).float().mean().item()
    assert seq_match == 1.0, seq_match
    assert _CALLS["n"] >= 5, _CALLS

    plugin_loss = model(ids).float().pow(2).mean()
    plugin_loss.backward()
    emb_grad_diff = (
        model.embedding.weight.grad.float()
        - baseline_embedding_grad.float()
    ).abs().max().item()
    head_grad_diff = (
        model.mlp[0].weight.grad.float()
        - baseline_head_grad.float()
    ).abs().max().item()
    assert emb_grad_diff < 2e-3, emb_grad_diff
    assert head_grad_diff < 2e-3, head_grad_diff

    print(
        f"  FlagOS stack: flag_gems.only_enable(['gelu']) active\n"
        f"  consumer: nn.Embedding + 2-layer MLP, bf16, 4 prompts × 48 tokens"
        f"\n  interception count={_CALLS['n']}; logits diff={logits_diff:.3e}; "
        f"greedy seq match={seq_match:.2f}"
        f"\n  embedding grad diff={emb_grad_diff:.3e}; "
        f"first-linear grad diff={head_grad_diff:.3e}"
    )
    return {
        "ok": True,
        "intercepted": _CALLS["n"],
        "logits_max_diff": logits_diff,
        "top1_rate": 1.0,
        "greedy_seq_match": seq_match,
        "embedding_grad_diff": round(emb_grad_diff, 8),
        "consumer": "nn.Embedding+TokenMLP",
    }


if __name__ == "__main__":
    from _profile import load_profile

    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
