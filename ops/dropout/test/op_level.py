# op 层验证: A1 aten 注册 → 拦截命中 → 确定性/随机分支不变形。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch
    import torch.nn.functional as F

    import register as reg

    dev = profile.torch_device

    reg.CALL_COUNT["dropout"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    # 1) 拦截: F.dropout 命中
    x = torch.randn(256, 128, dtype=torch.float16, device=dev)
    before = reg.CALL_COUNT["dropout"]
    _ = F.dropout(x, 0.0, True)
    assert reg.CALL_COUNT["dropout"] > before, "F.dropout 未被拦截"
    after = reg.CALL_COUNT["dropout"]

    # 2) 确定性分支
    assert F.dropout(x, 0.0, True) is x, "p=0 未别名"
    assert F.dropout(x, 0.5, False) is x, "train=False 未别名"
    z = F.dropout(x, 1.0, True)
    assert torch.equal(z, torch.zeros_like(x)), "p=1 非全零"
    assert reg.CALL_COUNT["dropout"] > after

    # 3) 随机分支: 结构 + 统计
    inp = torch.randn(64, 1024, dtype=torch.float32, device=dev) * 2
    torch.manual_seed(7)
    out = F.dropout(inp, 0.5, True)
    assert out.shape == inp.shape and out.data_ptr() != inp.data_ptr()
    keep = out != 0
    want = inp[keep].float() * 2.0
    assert ((out[keep].float() - want).abs() / (want.abs() + 1e-6)
            ).max().item() < 0.05, "存活元素缩放错误"
    drop = (~keep).float().mean().item()
    assert abs(drop - 0.5) < 0.02, f"drop 比例 {drop}"

    # 4) 非法 p
    try:
        F.dropout(x, 1.7, True)
        raise AssertionError("非法 p 未报错")
    except (RuntimeError, ValueError):
        pass

    return {"ok": True,
            "registered": f"aten::dropout@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["dropout"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
