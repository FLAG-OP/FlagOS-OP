# 语义参考实现——判卷标准，先于 kernel 编写。
#
# 约定:
#   - 内部用 fp32 计算，最后 cast 回输入 dtype
#   - 不依赖任何注册/分发（纯函数，kernel 层/应用层共用）
import torch


def my_op_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """<TODO: 一句话说清数学语义，如 'GELU(tanh 近似)(x) * gate'>。"""
    xf = x.float()
    # <TODO: fp32 参考实现>
    out = xf * gate.float()
    return out.to(x.dtype)
