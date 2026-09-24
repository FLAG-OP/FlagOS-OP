# op 层验证: A1 注册 → 拦截命中 → 元数据不变形 + 存储独立。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))


def _meta(t):
    return (tuple(t.shape), t.dtype, tuple(t.stride()))


def run(profile):
    import torch

    import register as reg

    dev = profile.torch_device

    # 注册前取原生参考元数据
    x = torch.randn(256, 128, dtype=torch.float32, device=dev).t()
    exp_meta = _meta(torch.empty_like(x))
    exp_mf = _meta(torch.empty_like(x, memory_format=torch.contiguous_format))

    reg.CALL_COUNT["empty_like"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    before = reg.CALL_COUNT["empty_like"]
    out = torch.empty_like(x)
    assert reg.CALL_COUNT["empty_like"] > before, "aten::empty_like 未被拦截"
    assert _meta(out) == exp_meta, f"元数据变形 {_meta(out)} != {exp_meta}"
    assert out.data_ptr() != x.data_ptr(), "共享存储"

    out2 = torch.empty_like(x, memory_format=torch.contiguous_format)
    assert _meta(out2) == exp_mf, "contiguous_format 元数据变形"
    assert out2.is_contiguous()

    # dtype/device 覆盖
    out3 = torch.empty_like(x, dtype=torch.float16)
    assert out3.dtype == torch.float16 and out3.shape == x.shape

    return {"ok": True,
            "registered": f"aten::empty_like@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["empty_like"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
