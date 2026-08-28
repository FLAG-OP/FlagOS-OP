# torch 级实现: 不产生新设备码，价值在融合/粘合与工程接入。
# 两种形态见本库样例:
#   - Python ATen 组合（本文件）→ examples/a1-framework 的恒等计数
#   - C++ 调 ATen（cpp_extension）→ examples/b-fullstack/fullstack_plugin.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def my_op_torch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """与 reference 同语义的 ATen 组合（fp32 内部计算）。"""
    xf = x.float()
    # <TODO: 用 ATen 算子组合出融合语义，例如>
    out = F.gelu(xf, approximate="tanh") * gate.float()
    return out.to(x.dtype)
