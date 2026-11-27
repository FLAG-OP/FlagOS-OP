# softmax 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor):
#   - 沿 dim 归一化: exp(x - max(x, dim)) / sum(exp(x - max), dim)
#   - dtype 参数: 输出 cast 到该 dtype（None = 保持 self.dtype）;
#     PyTorch 语义是"先在计算 dtype 下 softmax，再 cast"——内部升 fp32
#     计算是 CUDA/PPU 的实现行为，参考实现显式按 fp32 计算后 cast，
#     与 acceptance.md「参考实现约定」一致（fp32 内部精度 + cast 回）。
#   - 与 _softmax 的关系（issue #55 要求记录）:
#     aten::softmax(self, dim, dtype) 对半精度输入等价
#     _softmax(self, dim, half_to_float) 其中
#     half_to_float = (dtype 为 fp32 且 self 为 fp16/bf16)。
#     见本目录 PLATFORM.md「与 _softmax 的调用关系」。
#
# 不依赖任何注册/分发（纯函数，kernel 层/应用层共用）。
from __future__ import annotations

import torch


def softmax_reference(self: torch.Tensor, dim: int = -1,
                      dtype: torch.dtype | None = None) -> torch.Tensor:
    """aten::softmax 的 fp32 内部精度参考实现。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("softmax expects a Tensor")
    xf = self.float()
    out = torch.softmax(xf, dim=dim)
    return out.to(dtype if dtype is not None else self.dtype)
