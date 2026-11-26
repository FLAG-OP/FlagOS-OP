# torch 级实现: ATen 组合表达 softmax（不产生新设备码）。
#
# 用途:
#   - 三方对照「自研 Triton vs ATen 组合 vs 参考」中的组合侧
#   - 语义教学: softmax = exp(x - max) / sum(exp(x - max)) 的显式分解
#
# 递归规避（sdpa 教训）: 组合用的 ATen 原语选 max/exp/sum——**不调**
# torch.softmax / aten::softmax（A1 注册后重新进入本注册会无限递归）。
from __future__ import annotations

import torch


def softmax_torch(self: torch.Tensor, dim: int = -1,
                  dtype: torch.dtype | None = None) -> torch.Tensor:
    """ATen 组合 softmax: fp32 内部精度，输出 cast 回（同 reference 口径）。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("softmax expects a Tensor")
    xf = self.float()
    x_max = torch.amax(xf, dim=dim, keepdim=True)
    e = torch.exp(xf - x_max)
    out = e / torch.sum(e, dim=dim, keepdim=True)
    return out.to(dtype if dtype is not None else self.dtype)
