# 注册: `empty` **不做 A1 注册**（route = 自用/实验）。
#
# 原因: empty 是分配原语。若同时注册 empty 与 empty_strided，二者实现会
# 互相调用形成递归；FlagGems 也未在 _FULL_CONFIG 注册 aten::empty.memory_format
# （其源码中该项被注释）。本算子仅提供 torch 级语义与三层验证。
from __future__ import annotations

ROUTE = "自用/实验"


def register_a1(dispatch_key: str = "PrivateUse1"):
    raise NotImplementedError(
        "empty 不做 A1 注册：分配原语注册会与 empty_strided 递归，"
        "且全局覆盖 torch.empty 风险高。route=自用/实验。"
    )
