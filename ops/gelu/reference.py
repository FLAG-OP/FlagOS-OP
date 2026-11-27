# gelu 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::gelu(Tensor self, *, str approximate='none') -> Tensor):
#   - approximate='none'（精确/erf 形式）:
#       gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
#   - approximate='tanh'（TanH 近似形式, Hendrycks & Gimpel 2016 §附录）:
#       gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#   - 独立高精度: 内部按 fp64 计算、最后 cast 回输入 dtype（负尾
#     backward 低于 fp32 ULP 时仍可判卷; 不 cast 回去会把 bf16 自身的
#     量化噪声当成实现误差）。
#   - 逐元素算子，无形状约束；0 维张量与空张量合法。
#
# 不依赖任何注册/分发（纯函数，kernel 层/应用层共用）。
# 判卷依据 = 独立高精度参考（本文件 fp64）+ 原生 ATen 结果三方对拍
# （issue #55 交付要求 3）。
from __future__ import annotations

import math

import torch


def gelu_reference(self: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """aten::gelu 的 fp64 内部精度参考实现。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("gelu expects a Tensor")
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    x = self.double()
    if approximate == "none":
        y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    else:
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1.0 + torch.tanh(inner))
    return y.to(self.dtype)


def gelu_backward_reference(
    grad_output: torch.Tensor, self: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    """aten::gelu_backward 参考: d/dx gelu(x)，fp64 内部精度。

    erf 形式:  0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)
    tanh 形式: 代数展开见 kernel/triton_level.py 同式。
    """
    x = self.double()
    if approximate == "none":
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        d = cdf + x * pdf
    else:
        k = math.sqrt(2.0 / math.pi)
        inner = k * (x + 0.044715 * x * x * x)
        t = torch.tanh(inner)
        # d/dx [0.5x(1+tanh(u))] = 0.5(1+tanh u) + 0.5x·sech²(u)·u'
        sech2 = 1.0 - t * t
        du = k * (1.0 + 3.0 * 0.044715 * x * x)
        d = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du
    return (grad_output.double() * d).to(grad_output.dtype)
