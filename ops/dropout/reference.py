# dropout 语义参考实现。
#
# 语义 (aten::dropout(Tensor input, float p, bool train) -> Tensor):
#   - p ∉ [0,1] → RuntimeError（无论 train）。
#   - train=False 或 p==0 → 返回 input 本身（别名，不拷贝）。
#   - p==1（train=True）→ 返回全零，suggested memory format。
#   - 0<p<1 且 train=True → out = input * mask/(1-p)，mask ~ Bernoulli(1-p)。
#
# ⚠️ 判定口径: Triton 的 Philox 流与 PyTorch 原生 dropout 不同，**不能**
# 逐元素比较随机路径。本参考实现用于:
#   - 确定性分支（train=False / p=0 / p=1）的位级判卷；
#   - 随机分支（0<p<1）的结构/统计对照（scale、drop 比例、依赖 seed）。
from __future__ import annotations

import torch
import torch.nn.functional as F


def dropout_reference(input: torch.Tensor, p: float = 0.5,
                      train: bool = True) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("dropout expects a Tensor")
    return F.dropout(input, p, train)
