# 智能路由（auto impl）op 层验证: patch 安装/路由正确性/开关/性能/梯度
# 运行: python3 test/op_level_auto.py [ascend910]
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def _mk(S, dtype, dev, seed=5):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda n: (torch.randn(*n, generator=g) * 0.5).to(dtype).to(dev)
    return mk((1, 4, S, 64)), mk((1, 4, S, 64)), mk((1, 4, S, 64))


def run(profile):
    import time

    import torch
    import torch_npu  # noqa: F401

    from kernel.auto_dispatch import (reset_stats, stats, remove_patch)
    from register import register_a1

    dev = profile.torch_device
    F = torch.nn.functional

    # 1) 基线（未 patch）: 大小 S 原生参照
    qS, kS, vS = _mk(256, torch.float16, dev)
    qL, kL, vL = _mk(4096, torch.float16, dev)
    natS = F.scaled_dot_product_attention(qS, kS, vS, is_causal=True)
    natL = F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)

    # 2) auto 模式: install_patch（无 aten 注册, 无递归面）
    reset_stats()
    st = register_a1(counter={}, impl="auto")

    # 小 S → 自研 triton（数值容差内）
    outS = F.scaled_dot_product_attention(qS, kS, vS, is_causal=True)
    errS = (outS.float() - natS.float()).abs().max().item()
    # 大 S → 原始 F.sdpa 引用 = 原生（逐位一致）
    outL = F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
    bitwise = torch.equal(outL, natL)

    assert st["routed_triton"] >= 1, st
    assert st["routed_native"] >= 1, st
    assert errS < 2e-2, f"小S路由实现数值超差: {errS}"
    assert bitwise, "大S路由未命中原始 F.sdpa"

    # 3) 开关: 强制模式
    import os
    reset_stats()
    os.environ["SDPA_DISPATCH_MODE"] = "triton"
    F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
    assert st["forced_triton"] == 1, st
    os.environ["SDPA_DISPATCH_MODE"] = "native"
    outN = F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
    assert st["forced_native"] == 1 and torch.equal(outN, natL), st
    del os.environ["SDPA_DISPATCH_MODE"]

    # 4) 性能: auto vs 纯 triton（大 S, 经 patch 路径）
    from kernel.triton_level import sdpa_triton
    for _ in range(10):
        sdpa_triton(qL, kL, vL, None, 0.0, True, None, False)
        F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        F.scaled_dot_product_attention(qL, kL, vL, is_causal=True)
    torch.npu.synchronize()
    auto_ms = (time.perf_counter() - t0) / 30 * 1000
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        sdpa_triton(qL, kL, vL, None, 0.0, True, None, False)
    torch.npu.synchronize()
    triton_ms = (time.perf_counter() - t0) / 30 * 1000
    speedup = triton_ms / auto_ms
    assert speedup > 3.0, f"auto 大S加速不足: {speedup:.1f}x"

    # 5) 梯度: 小 S triton 路径 autograd 完整; 大 S 原生自带 autograd
    qg, kg, vg = _mk(512, torch.float16, dev, seed=9)
    for t in (qg, kg, vg):
        t.requires_grad_(True)
    out = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True)
    assert out.requires_grad
    out.float().sum().backward()
    assert qg.grad is not None and torch.isfinite(qg.grad).all()

    qg2, kg2, vg2 = _mk(4096, torch.float16, dev, seed=10)
    for t in (qg2, kg2, vg2):
        t.requires_grad_(True)
    out2 = F.scaled_dot_product_attention(qg2, kg2, vg2, is_causal=True)
    assert out2.requires_grad
    out2.float().sum().backward()
    assert qg2.grad is not None

    # 6) 还原 patch 幂等
    remove_patch()
    remove_patch()
    outR = F.scaled_dot_product_attention(qS, kS, vS, is_causal=True)
    assert torch.equal(outR, natS), "remove_patch 后未还原原始 F.sdpa"

    print(f"  路由: 小S→triton (err {errS:.2e}) / 大S→原始F.sdpa (逐位)")
    print(f"  开关: forced triton/native 各命中 ✓")
    print(f"  性能: auto {auto_ms:.3f}ms vs 纯triton {triton_ms:.3f}ms "
          f"= {speedup:.1f}x (S=4096)")
    print(f"  梯度: 双路径 autograd 完整 ✓ / patch 还原幂等 ✓")
    return {"ok": True, "route_small": "triton", "route_large": "native",
            "auto_ms_S4096": round(auto_ms, 3),
            "speedup_vs_pure_triton": round(speedup, 1)}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "ascend910")))
