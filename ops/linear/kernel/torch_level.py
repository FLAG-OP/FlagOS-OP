# torch 级实现: 不产生新设备码，用 ATen 组合表达 linear。
#
# 用途:
#   - 作为 torch 级候选参与黄金/性能对照（与 reference 同语义）
#   - A1 注册的 fallback 实现（原生 linear 不可用时组合兜底）
#
# 递归规避（sdpa 教训, PLATFORM.md §3c）: A1 注册后 backend 内**不得**
# 再调 F.linear / torch.ops.aten.linear（会重新进入本注册）。这里用
# matmul + add 组合落回原生 matmul 语义——若同进程亦注册了 ops.matmul
# 的 A1，则组合会路由到 matmul 实现（算子族行为, 预期且在测试中覆盖）。
from __future__ import annotations

import torch


def linear_torch(input: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor | None = None) -> torch.Tensor:
    """与 reference 同语义的 ATen 组合（matmul + add）。"""
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("linear expects Tensors")
    y = torch.matmul(input, weight.t())
    if bias is not None:
        y = y + bias
    return y
