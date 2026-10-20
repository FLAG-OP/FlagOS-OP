# empty_like 语义参考实现——判卷标准。
#
# 语义 (aten::empty_like(Tensor self, *, ScalarType? dtype=None,
#       Layout? layout=None, Device? device=None, bool? pin_memory=None,
#       MemoryFormat? memory_format=None) -> Tensor):
#   - 分配**未初始化**内存，shape 同 self；dtype/layout/device 默认继承 self。
#   - memory_format 默认 preserve_format（dense 保留 self.stride()，否则连续）。
#   - 只保证元数据（shape/dtype/stride/device），**不保证数值**（未初始化）。
from __future__ import annotations

import torch


def empty_like_reference(self: torch.Tensor, *, dtype=None, layout=None,
                         device=None, pin_memory=None, memory_format=None):
    """设备无关的元数据参考（判卷标准）。

    不走 `torch.empty_like`（其 preserve_format 行为在设备间不一致：CPU 保留
    dense stride，torch_npu 对非连续输入退化为连续——见 PLATFORM.md），改为按
    文档语义经 meta 设备推导 stride，保证 CPU/MLU/NPU 判卷标准一致。

    Dashener2 2026-10-15-fix（Ascend 移植）
    """
    if not isinstance(self, torch.Tensor):
        raise TypeError("empty_like expects a Tensor")
    dtype = self.dtype if dtype is None else dtype
    layout = self.layout if layout is None else layout
    device = self.device if device is None else device
    if memory_format is None:
        memory_format = torch.preserve_format
    if memory_format == torch.preserve_format:
        strides = tuple(self.to("meta").stride())
    else:
        strides = tuple(torch.empty(tuple(self.shape), dtype=dtype,
                                    device="meta",
                                    memory_format=memory_format).stride())
    return torch.empty_strided(tuple(self.shape), strides, dtype=dtype,
                               layout=layout, device=device,
                               pin_memory=pin_memory)
