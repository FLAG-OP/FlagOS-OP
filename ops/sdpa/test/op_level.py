# op 层验证: A1 aten 注册 → 拦截命中 → 精度不变形 → autograd 完整。
# 运行: python3 test/op_level.py [ascend910|p800-kunlunxin]
#
# 注: "撤销恢复"无法进程内验证（torch.library 注册不可撤销，实测
# torch 2.10），原生对照在注册前采集 + 独立进程复跑，见 test_op_phase2.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

_CALLS = {"n": 0}


def _make(dtype, dev, seed=3, requires_grad=False):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda n: (torch.randn(*n, generator=g) * 0.5)   # noqa: E731
    q = mk((1, 4, 128, 64)).to(dtype).to(dev).requires_grad_(requires_grad)
    k = mk((1, 4, 128, 64)).to(dtype).to(dev).requires_grad_(requires_grad)
    v = mk((1, 4, 128, 64)).to(dtype).to(dev).requires_grad_(requires_grad)
    return q, k, v


def run(profile):
    import torch
    if profile.torch_device.startswith("npu"):
        import torch_npu  # noqa: F401
    if profile.torch_device.startswith("mlu"):
        import torch_mlu  # noqa: F401

    from kernel.triton_level import sdpa_triton
    from register import register_a1

    dev = profile.torch_device
    F = torch.nn.functional

    # 1) 基线: 注册前（平台原生实现，缓存输出）
    q, k, v = _make(torch.float16, dev)
    out_native = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    direct_grad_check = "not-applicable"
    if profile.vendor in ("kunlunxin", "cambricon"):
        # 厂商/Triton 直调路径平台分化回归: q/k/v 需按需走 A1 数学 backward；
        # 可微 float mask 厂商路径不支持，应回退 A1 数学 backward。
        qd, kd, vd = _make(torch.float16, dev, requires_grad=True)
        out_d = sdpa_triton(qd, kd, vd, None, 0.0, True, None, False)
        out_d.float().sum().backward()
        assert torch.isfinite(qd.grad).all(), "direct q/k/v backward 失败"

        qm, km, vm = _make(torch.float16, dev, requires_grad=True)
        mask = torch.randn(
            (1, 4, 128, 128), dtype=torch.float16, device=dev,
            requires_grad=True,
        )
        out_m = sdpa_triton(qm, km, vm, mask, 0.0, True, None, False)
        out_m.float().sum().backward()
        assert torch.isfinite(qm.grad).all() and torch.isfinite(mask.grad).all(), \
            "direct float-mask backward 失败"
        direct_grad_check = "vendor+math-fallback"

    # 2) A1 注册（profile 提供 dispatch key）
    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)

    q, k, v = _make(torch.float16, dev)
    out_hooked = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _CALLS["n"] > 0, "拦截未命中"

    # 3) 注册后输出 == 直调输出（逐位）
    out_direct = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    assert torch.equal(out_hooked, out_direct), "注册路径 ≠ 直调路径"

    # 4) 与原生数值差（fp16 容差）
    err_native = (out_hooked.float() - out_native.float()
                  ).abs().max().item()
    assert err_native < 2e-2, f"与原生差异超容差: {err_native}"

    # 5) autograd: requires_grad 前向有 grad_fn，反向梯度 vs 原生梯度
    qg, kg, vg = _make(torch.float16, dev, requires_grad=True)
    out_g = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True)
    assert out_g.requires_grad, "autograd 被破坏（无 grad_fn）"
    loss = out_g.float().sum()
    loss.backward()
    dq_ours, dk_ours, dv_ours = qg.grad, kg.grad, vg.grad

    # 原生梯度（独立子进程，避免注册污染）
    ph2 = subprocess.run(
        [sys.executable, str(OP_DIR / "test" / "op_phase2_native.py"),
         profile.torch_device],
        capture_output=True, text=True, timeout=300)
    import json as _json
    line = [ln for ln in ph2.stdout.splitlines() if ln.startswith("NATIVE_GRAD")]
    assert line, f"子进程失败: {ph2.stdout[-300:]} {ph2.stderr[-300:]}"
    gerr = _json.loads(line[0].split(" ", 1)[1])
    tol = 2e-2
    for name, d_o, d_n in [("dq", dq_ours, gerr["dq"]),
                           ("dk", dk_ours, gerr["dk"]),
                           ("dv", dv_ours, gerr["dv"])]:
        e = (d_o.float().cpu() - torch.tensor(d_n)).abs().max().item()
        assert e < tol, f"{name} 梯度误差超容差: {e}"
        print(f"  grad {name}: max_err_vs_native={e:.3e}")

    print(f"  拦截命中 count={_CALLS['n']}; 注册=直调(逐位); "
          f"vs 原生 max_diff={err_native:.3e}; autograd 前反向 OK")
    return {"ok": True, "intercepted": _CALLS["n"],
            "bitwise_vs_direct": True,
            "max_diff_vs_native": round(err_native, 8),
            "grad_check": "vs-native-subprocess",
            "direct_grad_check": direct_grad_check}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else None)))
