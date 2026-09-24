# 注册: detach_ **不可 A1 注册**（实测 PrivateUse1 不命中，属 autograd key）。
from __future__ import annotations

ROUTE = "自用/实验"


def register_a1(dispatch_key="PrivateUse1"):
    raise NotImplementedError(
        "detach_ 不做 A1 注册：其为 autograd 层原地算子，PrivateUse1 IMPL 不命中。"
    )
