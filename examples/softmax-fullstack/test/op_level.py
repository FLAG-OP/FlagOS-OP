# 框架层验证: aten::_softmax 注册 → 拦截 → 拦截路径精度。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def run(profile):
    import torch
    import torch.nn.functional as F

    from register import CALLS, register_softmax

    register_softmax(profile.dispatch_key)
    x = torch.randn(128, 512, dtype=torch.bfloat16,
                    device=profile.torch_device)
    before = CALLS["softmax"]
    out = F.softmax(x, dim=-1)
    assert CALLS["softmax"] == before + 1, "未被拦截"
    ref = F.softmax(x.cpu().float(), dim=-1).to(x.device)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 1e-2
    return {"ok": True, "intercepted": CALLS["softmax"],
            "max_err": round(err, 8)}
