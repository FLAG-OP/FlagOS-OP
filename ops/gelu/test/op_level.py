# op 层验证: A1 aten 注册 → 拦截命中 → 精度不变形 + 自带反向正确 + 回退边界。
#
# 注册键口径: thead-ppu 算子本地 AutogradCUDA（sdpa/PLATFORM.md §3c 先例;
# known-issues #19.3: inference_mode 下 Autograd* 键零命中为平台边界，
# 本测试第 5 步如实断言该边界而非误报失败）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))                 # 引用同算子的 register/kernel
sys.path.insert(0, str(OP_DIR.parent.parent))   # 仓库根（common 等）


def run(profile):
    import torch

    import register as reg
    from reference import gelu_backward_reference, gelu_reference

    dev = profile.torch_device
    reg.CALL_COUNT["gelu"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    def assert_close(got, expected, dtype, tag):
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        err = (got.float() - expected.float()).abs().max().item()
        assert err < tol, f"{tag} err={err}"

    # 1) 拦截验证: torch 宿主调用必须命中我们的实现
    x = torch.randn(512, 512, dtype=torch.bfloat16, device=dev)
    before = reg.CALL_COUNT["gelu"]
    out = torch.ops.aten.gelu(x)
    after = reg.CALL_COUNT["gelu"]
    assert after > before, "aten::gelu 未被拦截"
    assert _CALLS["n"] > 0, "counter 未记入 op 层拦截"
    assert_close(out, gelu_reference(x), x.dtype, "拦截后精度不一致（none）")

    # 1b) F.gelu 方法式调用同样命中（含 tanh 模式 + 关键字参数）
    out2 = torch.nn.functional.gelu(x, approximate="tanh")
    assert reg.CALL_COUNT["gelu"] > after, "F.gelu(approximate=) 未命中"
    assert_close(out2, gelu_reference(x, "tanh"), x.dtype, "tanh 精度不一致")

    # 2) 自带反向: AutogradCUDA 注册替换内置 autograd 节点 →
    #    反向走 _GeluFn 解析 backward，与 CPU fp32 参考对拍
    #    （注意: aten::gelu 的 approximate 为 keyword-only, 直调必须
    #    approximate= 传参——真机 schema 实测, 见 probes/thead_dense_p0_probe）。
    for mode in ("none", "tanh"):
        xg = (torch.randn(64, 3072, dtype=torch.float32, device=dev) * 2
              ).requires_grad_(True)
        y = torch.ops.aten.gelu(xg, approximate=mode)
        (gx,) = torch.autograd.grad(y.sum(), xg)
        gx_ref = gelu_backward_reference(
            torch.ones_like(gx).cpu(), xg.detach().cpu(), mode).to(dev)
        err = (gx - gx_ref).abs().max().item()
        assert err < 1e-5, f"自带反向 {mode} err={err}"

    # 3) 非连续/奇异输入经注册路径回退原生，语义不变
    xnc = torch.randn(64, 64, dtype=torch.float32, device=dev).t()
    assert_close(torch.ops.aten.gelu(xnc), gelu_reference(xnc),
                 xnc.dtype, "非连续回退不一致")
    x0 = torch.randn((), dtype=torch.float16, device=dev)  # 0 维
    assert_close(torch.ops.aten.gelu(x0), gelu_reference(x0),
                 x0.dtype, "0 维回退不一致")

    # 4) 半精度容差内对拍（fp16/bf16 三 dtype 组合）
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        a = torch.randn(1024, 512, dtype=torch.float32, device=dev).to(dt) * 3
        got = torch.ops.aten.gelu(a, approximate="none")
        exp = gelu_reference(a, "none")
        tol = 1e-5 if dt == torch.float32 else 1e-2
        err = (got.float() - exp.float()).abs().max().item()
        assert err < tol, f"{dt} err={err}"

    # 5) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3, sdpa/PLATFORM.md §3c 同款记录）
    with torch.inference_mode():
        xi = torch.randn(64, 64, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["gelu"]
        oi = torch.ops.aten.gelu(xi)
        hits_i = reg.CALL_COUNT["gelu"] - before_i
    assert_close(oi, gelu_reference(xi), xi.dtype,
                 "inference_mode 结果不一致")
    note = f"inference_mode 命中 {hits_i}（known-issues #19.3 边界）"

    return {"ok": True,
            "registered": f"aten::gelu@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["gelu"],
            "inference_mode_note": note}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
