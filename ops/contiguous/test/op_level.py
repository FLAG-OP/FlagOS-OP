# op 层验证: A1 aten 注册 → 拦截命中 → 语义/别名不变形 + memory_format 回退。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import contiguous_reference

    dev = profile.torch_device

    # 注册前取原生参考
    x = torch.randn(256, 128, dtype=torch.float32, device=dev).t()
    exp = contiguous_reference(x)
    cl = torch.randn(2, 8, 16, 16, device=dev)          # NCHW 连续 → 需转 channels_last
    cl_ref = contiguous_reference(cl, memory_format=torch.channels_last)

    reg.CALL_COUNT["contiguous"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    # 1) 拦截验证（非连续输入）
    before = reg.CALL_COUNT["contiguous"]
    out = x.contiguous()
    assert reg.CALL_COUNT["contiguous"] > before, "aten::contiguous 未被拦截"
    assert out.is_contiguous() and torch.equal(out, exp)

    # 2) 别名快路径
    c = torch.randn(64, 64, dtype=torch.bfloat16, device=dev)
    assert c.contiguous() is c, "已连续输入未返回 self"

    # 3) 多 dtype 拷贝路径
    for dt in (torch.float16, torch.float32, torch.bfloat16):
        a = torch.randn(128, 64, dtype=dt, device=dev).t()
        b = a.contiguous()
        assert b.is_contiguous() and torch.equal(b, a)

    # 4) memory_format 回退（channels_last）
    cl_out = cl.contiguous(memory_format=torch.channels_last)
    assert cl_out.stride() == cl_ref.stride(), "channels_last 回退 stride 不一致"
    assert torch.equal(cl_out, cl_ref)

    return {"ok": True,
            "registered": f"aten::contiguous@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["contiguous"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
