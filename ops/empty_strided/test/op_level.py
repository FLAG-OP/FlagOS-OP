# op 层验证: A1 注册 → 拦截命中 → stride 元数据不变形。
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
    # 注册前取原生元数据（stride 与设备无关）
    exp = _meta(torch.empty_strided((128, 256), (1, 128),
                                    dtype=torch.float32, device=dev))

    reg.CALL_COUNT["empty_strided"] = 0
    lib = reg.register_a1(profile.dispatch_key)
    assert lib is not None

    before = reg.CALL_COUNT["empty_strided"]
    out = torch.empty_strided((128, 256), (1, 128), dtype=torch.float32,
                              device=dev)
    assert reg.CALL_COUNT["empty_strided"] > before, "未被拦截"
    assert _meta(out) == exp, f"元数据变形 {_meta(out)} != {exp}"

    # 重叠 stride
    ov = torch.empty_strided((4, 4), (1, 0), dtype=torch.float32, device=dev)
    assert ov.stride() == (1, 0)

    return {"ok": True,
            "registered": f"aten::empty_strided@{profile.dispatch_key}",
            "intercepted_calls": reg.CALL_COUNT["empty_strided"]}


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "cambricon")))
