#!/usr/bin/env python3
"""样例 B×算子层: 自定义厂商算子注册/选择/计数。

自包含演示:
  ① 定义厂商 Backend（vendor 属性 / is_available / 算子方法）
  ② 以 VENDOR 类型注册进 dispatch registry
  ③ with_allowed_vendors 精确选择 + 调用计数 + 语义精度

运行: python3 examples/b-op/example.py [设备profile名]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch


# ============ ① 厂商 Backend 定义 ============
def register_builtins(registry) -> None:
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
    from vllm_fl.dispatch.backends.base import Backend

    COUNT_FILE = os.environ.get("MYVENDOR_COUNT_FILE", "/tmp/myvendor_counts.json")

    def bump(op):
        counts = {}
        try:
            counts = json.load(open(COUNT_FILE))
        except (OSError, ValueError):
            pass
        counts[op] = counts.get(op, 0) + 1
        json.dump(counts, open(COUNT_FILE, "w"))

    class MyVendorBackend(Backend):
        """演示 vendor: 生产中方法体调用厂商库 kernel。"""

        @property
        def name(self): return "myvendor"

        @property
        def vendor(self): return "myvendor"   # vendor 路线必须非 None

        def is_available(self):
            # 生产中检测厂商硬件/库: import my_vendor_lib
            return True

        def silu_and_mul(self, obj, x):
            bump("silu_and_mul")
            # 生产替换为: import xtorch_ops; xtorch_ops.swiglu(x, out)
            d = x.shape[-1] // 2
            return (torch.nn.functional.silu(x[..., :d].float())
                    * x[..., d:].float()).to(x.dtype)

    backend = MyVendorBackend()
    registry.register_many([
        OpImpl(op_name="silu_and_mul", impl_id="vendor.myvendor",
               kind=BackendImplKind.VENDOR, fn=backend.silu_and_mul,
               vendor="myvendor", priority=BackendPriority.VENDOR),
    ])


register = register_builtins
vllm_fl_register = register_builtins


def main() -> None:
    from common.device import load_profile

    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    dev = profile.torch_device
    os.environ["VLLM_FL_PLUGIN_MODULES"] = __name__
    os.environ["MYVENDOR_COUNT_FILE"] = cnt = "/tmp/myvendor_example_counts.json"
    json.dump({}, open(cnt, "w"))
    print(f"[b-op 样例] 设备: {profile.summary()}")

    # ============ ②/③ 注册 + 选择 + 计数 + 精度 ============
    from vllm_fl.dispatch import get_default_manager, call_op
    from vllm_fl.dispatch.policy import with_preference, with_allowed_vendors

    m = get_default_manager()
    m.ensure_initialized()
    ids = [i.impl_id for i in
           m.registry.snapshot().impls_by_op.get("silu_and_mul", [])]
    assert "vendor.myvendor" in ids, ids
    print(f"  注册: {ids}")

    x = torch.randn(64, 8192, dtype=torch.bfloat16, device=dev)
    d = x.shape[-1] // 2
    ref = (torch.nn.functional.silu(x[..., :d].float())
           * x[..., d:].float()).to(x.dtype)

    with with_preference("vendor"), with_allowed_vendors("myvendor"):
        out = call_op("silu_and_mul", None, x)
        used = m._called_ops["silu_and_mul"]
    assert used == "vendor.myvendor", used
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 1e-1, err

    counts = json.load(open(cnt))
    print(f"  选择: {used}（with_allowed_vendors 精确钉住）")
    print(f"  精度: max_err={err:.3e}")
    print(f"  计数: {counts}")
    print("=> B×op 样例 PASS")


if __name__ == "__main__":
    main()
