# op 层验证: _softmax 原语 A1 注册 → 拦截 → 与公开 softmax 的关系取证。
#
# issue #55: softmax 与 _softmax 必须同时验证, 记录调用关系与注册边界:
#   - 本目录注册原语 aten::_softmax;
#   - 公开 aten::softmax 的注册在 ops/softmax（不同算子名, 互不遮蔽）;
#   - 单独注册原语时, 公开 API 的内部落点是否消费原语注册（原生路径下
#     公开实现直接调原语 kernel, 不再经 Python 注册面）→ 如实取证记录。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def run(profile):
    import torch

    import register as reg
    from reference import softmax_prim_reference

    dev = profile.torch_device
    reg.CALL_COUNT["_softmax"] = 0
    _CALLS = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=_CALLS)
    assert lib is not None

    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev) * 8

    # 1) 原语直调拦截
    before = reg.CALL_COUNT["_softmax"]
    out = torch.ops.aten._softmax(x, -1, False)
    assert reg.CALL_COUNT["_softmax"] > before, "aten::_softmax 未被拦截"
    assert _CALLS["n"] > 0
    assert (out.float() - softmax_prim_reference(x, -1).to(dev).float()
            ).abs().max().item() < 1e-2

    # 1b) half_to_float=True 直调
    xh = torch.randn(32, 512, dtype=torch.float16, device=dev) * 8
    goth = torch.ops.aten._softmax(xh, -1, True)
    assert goth.dtype == torch.float32
    assert (goth - softmax_prim_reference(xh, -1, True).to(dev)
            ).abs().max().item() < 2e-5

    # 2) 调用关系取证: 单独注册原语后, 公开 F.softmax 的命中情况
    #    （原生实现内部直接调原语 kernel, 不经 Python 注册面——计数不涨
    #    属预期; 与 ops/softmax 的公开注册面互补归因, 详见 README）。
    before2 = reg.CALL_COUNT["_softmax"]
    _ = torch.nn.functional.softmax(x, -1)
    public_hits = reg.CALL_COUNT["_softmax"] - before2

    # 3) 梯度链路（原生 SoftmaxBackward 保留于基键之上）
    xg = (torch.randn(32, 256, dtype=torch.float32, device=dev) * 4
          ).requires_grad_(True)
    torch.ops.aten._softmax(xg, -1, False).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()

    # 4) 平台边界（如实登记, 不算失败）: inference_mode 下 AutogradCUDA
    #    零命中（known-issues #19.3）
    with torch.inference_mode():
        xi = torch.randn(16, 256, dtype=torch.float16, device=dev)
        before_i = reg.CALL_COUNT["_softmax"]
        torch.ops.aten._softmax(xi, -1, False)
        hits_i = reg.CALL_COUNT["_softmax"] - before_i

    return {"ok": True,
            "registered": f"aten::_softmax@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["_softmax"],
            "public_softmax_hits_primitive_registry": public_hits,
            "inference_mode_note": f"inference_mode 命中 {hits_i}"}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
