"""<算子名> platform backends（第二平台出现时启用本结构）。

范式来源: ops/sdpa/kernel/backends/（三平台实证）。单平台阶段可
只保留 triton_level.py 顶层实现; 当第二平台的平台绑定清单
（PLATFORM.md §2）与首个平台结构性分叉时, 按此拆分:

  kernel/
  ├── triton_level.py      # 公共入口保持不变（facade, tests 兼容）
  └── backends/
      ├── __init__.py      # 本文件: 设备类型 → backend 选择器
      ├── ascend910.py     # 各平台实现（含 PLATFORM 元数据 + 调用守卫）
      └── <platform2>.py

wt <wangt635@ustc.edu.cn>
"""
from __future__ import annotations

import importlib

# torch device.type → backend 模块名。
# 注意: 昆仑芯 XMLIR 将 XPU 呈现为 CUDA 张量（backend 内部自做
# torch_xmlir 守卫）; 寒武纪为 "mlu"。
_IMPL_BY_DEVICE = {
    "npu": "ascend910",
    # "cuda": "<vendor2>",
    # "mlu": "<vendor3>",
}


def get_impl(device_type: str):
    """按设备类型返回 backend 模块（惰性导入, 单例语义）。"""
    module_name = _IMPL_BY_DEVICE.get(device_type)
    if module_name is None:
        raise RuntimeError(
            f"<算子名> 无 {device_type!r} 平台实现，已注册: "
            f"{list(_IMPL_BY_DEVICE)}（见 PLATFORM.md / MERGE.md）")
    return importlib.import_module(f".{module_name}", __package__)
