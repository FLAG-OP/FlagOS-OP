# matmul 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::matmul(Tensor self, Tensor other) -> Tensor):
#   - 2D × 2D: 矩阵乘 (M,K)@(K,N)
#   - 1D × 2D / 2D × 1D: 向量-矩阵 / 矩阵-向量（降维输出）
#   - ≥3D: 批维广播乘（(B,M,K)@(K,N)、(B,M,K)@(B,K,N)、广播批维）
#   - 判卷: GEMM 形态 → 5e-2 rel（docs/acceptance.md 唯一权威表）
#
# 参考实现约定: 内部 fp32 计算、cast 回输入 dtype（acceptance.md）。
# 不依赖任何注册/分发（纯函数）。
from __future__ import annotations

import torch


def matmul_reference(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """aten::matmul 的 fp32 内部精度参考实现。"""
    if not isinstance(self, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise TypeError("matmul expects two Tensors")
    out_dtype = torch.result_type(self, other)
    return (self.float() @ other.float()).to(out_dtype)
