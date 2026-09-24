# op 层验证: A1 aten 注册 → 拦截命中 → 原地/广播/cast 不变形 + 回退。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import copy_reference

    dev = profile.torch_device

    # 注册前取原生参考
    dst = torch.randn(256, 128, dtype=torch.float32, device=dev)
    src = torch.randn(256, 128, dtype=torch.float16, device=dev)
    dst_ref = dst.clone()
    copy_reference(dst_ref, src)

    # 广播参考
    bdst = torch.randn(8, 64, 128, dtype=torch.bfloat16, device=dev)
    bsrc = torch.randn(64, 1, dtype=torch.bfloat16, device=dev)
    bdst_ref = bdst.clone()
    copy_reference(bdst_ref, bsrc)

    # int32 回退参考
    idst = torch.zeros(32, 32, dtype=torch.int32, device=dev)
    isrc = torch.randint(-100, 100, (32, 32), dtype=torch.int32, device=dev)
    idst_ref = idst.clone()
    copy_reference(idst_ref, isrc)

    reg.CALL_COUNT["copy_"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    # 1) 拦截 + 原地返回 + cast
    before = reg.CALL_COUNT["copy_"]
    out = dst.copy_(src)
    assert reg.CALL_COUNT["copy_"] > before, "aten::copy_ 未被拦截"
    assert out is dst, "copy_ 未返回 self"
    assert out.dtype == torch.float32
    assert torch.equal(out, dst_ref), "拦截后 cast 精度不一致"

    # 2) 广播
    bo = bdst.copy_(bsrc)
    assert bo is bdst and torch.equal(bdst, bdst_ref), "广播语义不一致"

    # 3) 回退: int32
    io = idst.copy_(isrc)
    assert io is idst and torch.equal(idst, idst_ref), "int32 回退不一致"

    # 4) 非 blocking 参数透传
    d2 = torch.zeros(64, 64, dtype=torch.float32, device=dev)
    s2 = torch.randn(64, 64, dtype=torch.float32, device=dev)
    d2.copy_(s2, non_blocking=True)
    assert torch.equal(d2, s2)

    return {"ok": True,
            "registered": f"aten::copy_@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["copy_"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
