"""op 层：A1 注册、aten 拦截、形状保持与 dtype 回退验证。"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def run(profile):
    import torch

    import importlib
    reg = importlib.import_module("ops.sigmoid.register")
    from ops.sigmoid.reference import sigmoid_reference

    device = profile.torch_device
    x = torch.linspace(-12, 12, 65535, dtype=torch.float32,
                       device=device).reshape(257, 255)
    expected = sigmoid_reference(x)

    fallback_inputs = {
        "int32": torch.arange(-32, 32, dtype=torch.int32, device=device),
        "int64": torch.arange(-32, 32, dtype=torch.int64, device=device),
        "bool": torch.tensor([False, True], dtype=torch.bool, device=device),
    }
    # 注册前取原生结果；注册后 torch.sigmoid 会命中自研实现。
    fallback_expected = {
        name: torch.sigmoid(value) for name, value in fallback_inputs.items()
    }

    reg.CALL_COUNT["sigmoid"] = 0
    counter = {"n": 0}
    lib = reg.register_a1(profile.dispatch_key, counter=counter,
                          platform=profile.name)
    assert lib is not None

    before = reg.CALL_COUNT["sigmoid"]
    out = torch.sigmoid(x)
    assert reg.CALL_COUNT["sigmoid"] > before, "aten::sigmoid 未被拦截"
    assert counter["n"] > 0, "A1 counter 契约未记录命中"
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)

    # 三种优化 dtype + 真实三维激活。
    for dtype, atol in ((torch.float16, 1e-3),
                        (torch.bfloat16, 8e-3),
                        (torch.float32, 2e-6)):
        sample = torch.randn(2, 64, 4096, dtype=dtype, device=device) * 8
        torch.testing.assert_close(sample.sigmoid(), sigmoid_reference(sample),
                                   atol=atol, rtol=max(atol, 2e-5))

    # 非连续和空张量保持 shape/dtype。
    view = torch.randn(129, 67, device=device).t()
    view_out = torch.sigmoid(view)
    torch.testing.assert_close(view_out, sigmoid_reference(view),
                               atol=2e-6, rtol=2e-5)
    empty = torch.empty(0, 4096, dtype=torch.float16, device=device)
    empty_out = torch.sigmoid(empty)
    assert empty_out.shape == empty.shape and empty_out.dtype == empty.dtype

    # int32/int64/bool 不走自研计算，但 A1 路径必须安全回退且语义一致。
    for name, value in fallback_inputs.items():
        fallback_out = torch.sigmoid(value)
        assert fallback_out.dtype == fallback_expected[name].dtype
        torch.testing.assert_close(fallback_out, fallback_expected[name])

    return {
        "ok": True,
        "registered": f"aten::sigmoid@{profile.dispatch_key}",
        "intercepted_calls": reg.CALL_COUNT["sigmoid"],
        "contract_counter": counter["n"],
        "optimized_dtypes": ["float16", "bfloat16", "float32"],
        "fallback_dtypes": list(fallback_inputs),
        "layout": "contiguous+noncontiguous+empty",
    }


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
