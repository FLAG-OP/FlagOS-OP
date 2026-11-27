# native_layer_norm 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::native_layer_norm(input, normalized_shape, weight, bias, eps)
#        -> (Tensor out, Tensor mean, Tensor rstd)):
#   - native_layer_norm 是 layer_norm 的底层原语（公开 API layer_norm 包装
#     它并丢弃 mean/rstd）; issue #55 要求两者**同时验证**（只覆盖公开
#     API 会遗漏底层路径）。
#   - out: (input - mean) / sqrt(var + eps) * weight + bias，按最后
#     len(normalized_shape) 维归一化; weight/bias 可空。
#   - mean/rstd: 归一化维上的均值与**标准差倒数**（1/sqrt(var+eps)），
#     形状为 input 形状去掉 normalized_shape 后缀。
#   - fp32/等变: 均值方差按 Welford/两趟在 fp32 累计（参考实现用
#     torch.mean/var float32 表达，与 ATen float 官方语义对拍）。
#
# 参考实现约定: 内部 fp32 计算、cast 回输入 dtype（acceptance.md）。
# 不依赖任何注册/分发（纯函数）。
from __future__ import annotations

import torch


def native_layer_norm_reference(input: torch.Tensor,
                                normalized_shape, weight, bias,
                                eps: float = 1e-5):
    """aten::native_layer_norm 参考: 返回 (out, mean, rstd)，fp32 内部精度。

    mean/rstd 语义 = 归一化维上的均值与 1/sqrt(var+eps)；形状约定为
    input 去掉 normalized_shape 后缀（CPU ATen 口径）。平台分化记录:
    PPU 原生返回 keepdim 形状（如 (16,1)）且恒 fp32（半精度输入亦然）
    ——值广播等价、形状平台分化，测试按 reshape 对拍（见 PLATFORM.md）。
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("native_layer_norm expects a Tensor")
    dims = [-i for i in range(1, len(normalized_shape) + 1)]
    out_dtype = input.dtype
    xf = input.float()
    mean = xf.mean(dim=dims, keepdim=True)
    var = xf.var(dim=dims, keepdim=True, correction=0)  # 有偏方差（ATen 口径）
    rstd = 1.0 / torch.sqrt(var + eps)
    y = (xf - mean) * rstd
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    red_shape = tuple(xf.shape[: len(xf.shape) - len(normalized_shape)])
    return (y.to(out_dtype),
            mean.reshape(red_shape),      # fp32 保持（统计量判卷口径）
            rstd.reshape(red_shape))
