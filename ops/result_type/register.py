# 注册: result_type **不可 A1 注册**（纯函数，非 backend dispatch）。
from __future__ import annotations

ROUTE = "自用/实验"


def register_a1(dispatch_key="PrivateUse1"):
    raise NotImplementedError(
        "result_type 不做 A1 注册：纯 dtype 元数据函数，PrivateUse1 IMPL 不命中。"
    )
