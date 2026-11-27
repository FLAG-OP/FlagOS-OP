# spmm 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::_sparse_mm(Tensor self, Tensor dense) -> Tensor; torch.spmm
# 是 torch.mm 的别名 → aten::mm, torch.sparse.mm 默认入口绑定
# aten::_sparse_mm——见 issue #57 实现范围 1 的绑定取证):
#   - self: CSR 稀疏矩阵 (M, K); dense: (K, N) 稠密矩阵; 输出 (M, N)。
#   - out[i, j] = Σ_k self[i, k] * dense[k, j]（仅非零 k 参与）。
#   - CSR 边界语义（issue #57 实现范围 2, 参考实现全部覆盖）:
#     空行 → 输出全零行; 重复列索引 → 值**累加**（CSR 数学定义）;
#     无序列索引 → 与顺序无关（求和可交换）。
#   - 梯度: dA_vals[r] = Σ_j g[row_r, j] * B[col_r, j];
#     dB = Aᵀ @ g（以 COO 形式 index_add）。
#
# 参考实现约定: 纯 CPU/torch 组合（index_add 原语），fp32 判卷
# （稀疏累加无融合精度问题; fp16/bf16 档按 acceptance.md 1e-2）。
# 不依赖任何注册/分发（纯函数）。
from __future__ import annotations

import torch


def spmm_reference(csr: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """CSR × dense 的参考实现（**fp64 独立高精度判卷标准**）。

    累加在 fp64 域进行、最后 cast 回值 dtype: SpMM 是乘加放大形态,
    大 K（如 4097×0.5 密度 ≈ 2000 非零/行）下 bf16/fp32 顺序累加的
    序噪声（实测 bf16 0.0625/rel 0.0996）会污染判卷——fp64 参考是
    独立标准, 自研 fp32 累加实现在 5e-2 rel（acceptance GEMM 档）
    内即 PASS。"""
    if not csr.is_sparse_csr:
        raise TypeError("spmm_reference expects a CSR sparse tensor")
    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()
    m = crow.shape[0] - 1
    n = dense.shape[1]
    rows = torch.repeat_interleave(
        torch.arange(m, device=val.device), crow[1:] - crow[:-1])
    contrib = val.double().unsqueeze(1) * dense.double().index_select(0, col)
    out = torch.zeros(m, n, dtype=torch.float64, device=val.device)
    out.index_add_(0, rows, contrib)
    return out.to(val.dtype)


def spmm_backward_reference(csr: torch.Tensor, dense: torch.Tensor,
                            grad_out: torch.Tensor):
    """SpMM 反向参考: (dA_vals, dB)——fp64 累加（口径同 spmm_reference）;
    dA 以 values 张量返回（稀疏结构视为常量, torch 稀疏 autograd 语义）。"""
    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()
    m = crow.shape[0] - 1
    rows = torch.repeat_interleave(
        torch.arange(m, device=val.device), crow[1:] - crow[:-1])
    # dA_vals[r] = Σ_j grad_out[row_r, j] * dense[col_r, j]
    d_vals = (grad_out.double().index_select(0, rows)
              * dense.double().index_select(0, col)).sum(-1)
    # dB = Aᵀ @ g: dB[k, j] += A[i, k] * g[i, j]
    dB = torch.zeros(dense.shape, dtype=torch.float64, device=dense.device)
    dB.index_add_(0, col,
                  val.double().unsqueeze(1)
                  * grad_out.double().index_select(0, rows))
    return d_vals.to(val.dtype), dB.to(dense.dtype)
