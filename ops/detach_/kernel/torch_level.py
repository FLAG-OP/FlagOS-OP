# torch 级实现: detach_ 不可 A1 注册（route=自用/实验），
# 故直接用原生 detach_（不会递归，因为没有注册）。
from __future__ import annotations


def detach__torch(self):
    return self.detach_()
