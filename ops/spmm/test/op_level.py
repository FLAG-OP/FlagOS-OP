# op 层验证: aten::_sparse_mm 注册 → 拦截 → 精度/反向/边界 + 稠密 mm 不受扰。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import spmm_reference

    dev = profile.torch_device
    reg.CALL_COUNT["_sparse_mm"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    m, k, n = 64, 32, 64   # 台账 N=64 对照尺寸
    dense = torch.rand(m, k, device=dev) * (
        torch.rand(m, k, device=dev) < 0.3)
    a = dense.to_sparse_csr()
    b = torch.randn(k, n, dtype=torch.float32, device=dev)

    # 1) 拦截验证: torch.sparse.mm（→ aten::_sparse_mm）命中
    before = reg.CALL_COUNT["_sparse_mm"]
    out = torch.sparse.mm(a, b)
    assert reg.CALL_COUNT["_sparse_mm"] > before, "aten::_sparse_mm 未被拦截"
    assert _CALLS["n"] > 0
    exp = spmm_reference(a.cpu(), b.cpu()).to(dev)
    err = (out.float() - exp.float()).abs().max().item()
    assert err < 1e-4, f"拦截后精度 err={err}"

    # 1b) 原语直调同样命中
    before2 = reg.CALL_COUNT["_sparse_mm"]
    torch.ops.aten._sparse_mm(a, b)
    assert reg.CALL_COUNT["_sparse_mm"] > before2, "_sparse_mm 直调未命中"

    # 2) 稠密 mm 别名不受扰（issue #57: 保留稠密 mm 现有行为）
    a2 = torch.randn(16, 32, dtype=torch.float32, device=dev)
    b2 = torch.randn(32, 8, dtype=torch.float32, device=dev)
    before3 = reg.CALL_COUNT["_sparse_mm"]
    dense_out = torch.mm(a2, b2)
    assert reg.CALL_COUNT["_sparse_mm"] == before3, "稠密 mm 被误拦截"
    ref = a2 @ b2
    assert (dense_out - ref).abs().max().item() < 1e-5, "稠密 mm 行为改变"

    # 3) 反向链路（自建 Function vs CPU 稀疏 autograd 参考）
    b3 = torch.randn(32, 16, dtype=torch.float32, device=dev,
                     requires_grad=True)
    out3 = torch.sparse.mm(a, b3)
    out3.sum().backward()
    assert b3.grad is not None and torch.isfinite(b3.grad).all()
    # CPU 参考梯度（dB = Aᵀ @ ones）。注意: CSR 的 .t() 返回 CSC（layout
    # 翻转, 平台/ATen 语义）——须 .to_sparse_csr() 转回再喂 reference。
    g = torch.ones_like(out3).cpu()
    aT = a.cpu().t().to_sparse_csr()
    rB = spmm_reference(aT, g)
    err = (b3.grad.cpu() - rB).abs().max().item()
    assert err < 1e-4, f"反向 dB err={err}"

    # 4) fp16/bf16 容差
    for dt in (torch.float16, torch.bfloat16):
        a16 = dense.to(dt).to_sparse_csr()
        b16 = b.to(dt)
        got = torch.sparse.mm(a16, b16)
        exp16 = spmm_reference(a16.cpu(), b16.cpu()).to(dev)
        e = (got.float() - exp16.float()).abs().max().item()
        assert e < 1e-2, f"{dt} err={e}"

    # 5) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        before_i = reg.CALL_COUNT["_sparse_mm"]
        torch.ops.aten._sparse_mm(a, b)
        hits_i = reg.CALL_COUNT["_sparse_mm"] - before_i
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::_sparse_mm@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["_sparse_mm"],
            "dense_mm_untouched": True,
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
