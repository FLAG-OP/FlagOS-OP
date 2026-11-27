# linear 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor):
#   - input: (..., in_features) 任意批维; weight: (out_features, in_features);
#     bias: (out_features,) 可空。
#   - 输出: input @ weight.T (+ bias)，批维广播。
#   - 无 bias 时等价 matmul(input, weight.t())；有 bias 时 2D 输入等价
#     addmm(bias, input, weight.t())（ATen 对 2D 走 addmm 融合路径，
#     但语义上 bias 加法与 matmul 可分离——fp32 参考按分离式计算，
#     判卷容差按 GEMM 档（docs/acceptance.md: 5e-2 rel）。
#
# 参考实现约定（docs/acceptance.md）: 内部 fp32 计算、最后 cast 回输入
# dtype——不 cast 会把 bf16 量化噪声当成实现误差。
# GEMM 用相对误差判卷（输出量级随输入放大，abs 不可比）。
#
# 不依赖任何注册/分发（纯函数，kernel 层/应用层共用）。
from __future__ import annotations

import torch


def linear_reference(input: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    """aten::linear 的 fp32 内部精度参考实现。"""
    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("linear expects Tensors")
    out_dtype = input.dtype
    xf = input.float()
    wf = weight.float()
    y = xf @ wf.t()
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype)
