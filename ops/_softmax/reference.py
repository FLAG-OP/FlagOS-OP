# _softmax 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor):
#   - softmax 的私有原语; 公开 aten::softmax(self, dim, dtype) 对半精度
#     输入等价 _softmax(self, dim, half_to_float) 其中
#     half_to_float = (dtype 为 fp32 且 self 为 fp16/bf16)——调用关系
#     记录见 ops/softmax/PLATFORM.md 与本目录 README.md（issue #55 要求
#     两者同时验证并记录注册边界）。
#   - half_to_float=True: 输出 fp32（计算亦在 fp32）; False: 输出同 dtype。
#   - 数值稳定: 减 max 后取指数; 输出有界 [0,1]，判卷 abs 容差
#     （fp32=1e-5, bf16/fp16=1e-2, docs/acceptance.md 唯一权威表）。
#
# 平台取证（真机 probes/thead_dense_p0_probe.py, 卡6, 2026-11-25）:
#   _softmax 直调 err=0.0; half_to_float=True 输出 fp32, err 1.8e-7。
# 不依赖任何注册/分发（纯函数）。
from __future__ import annotations

import torch


def softmax_prim_reference(self: torch.Tensor, dim: int,
                           half_to_float: bool = False) -> torch.Tensor:
    """aten::_softmax 的 fp32 内部精度参考实现。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("_softmax expects a Tensor")
    out_dtype = torch.float32 if half_to_float else self.dtype
    xf = self.float()
    m = xf.max(dim=dim, keepdim=True).values
    e = torch.exp(xf - m)
    y = e / e.sum(dim=dim, keepdim=True)
    return y.to(out_dtype)
