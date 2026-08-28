# op 层验证骨架: 注册 → 钉选命中 → 精度不变形。
from __future__ import annotations


def run(profile):
    import torch

    dev = profile.torch_device
    # <TODO: 按所选路线初始化注册>
    #   A1: register_a1(profile.dispatch_key)
    #   A2: VLLM_FL_PLUGIN_MODULES 指向本模块，或直接构造 registry

    x = torch.randn(512, 512, dtype=torch.bfloat16, device=dev) * 2
    g = torch.randn_like(x)

    # 钉选验证（A2/B）:
    # from vllm_fl.dispatch import call_op, get_default_manager
    # from vllm_fl.dispatch.policy import with_preference, with_allowed_vendors
    # with with_preference("vendor"), with_allowed_vendors("myvendor"):
    #     out = call_op("my_op", None, x, g)
    #     used = get_default_manager()._called_ops["my_op"]
    # assert used == "vendor.myvendor", used

    # 拦截验证（A1）: 调 torch 宿主算子，断言计数增加
    return {"ok": True, "registered": "<TODO>", "pinned": "<TODO>"}
