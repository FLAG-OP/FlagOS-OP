# copy_ 语义参考实现——判卷标准，先于 kernel 编写。
#
# 语义 (aten::copy_(Tensor self, Tensor src, bool non_blocking=False) -> Tensor):
#   - **原地**把 src 的元素写入 self，返回 self 本身。
#   - src 可广播到 self.shape；元素按 self.dtype cast。
#   - 纯搬运，无算术，原生 copy_ 即位级判卷标准。
#
# 注意: 会修改 dst，调用方需自行拷贝 dst（gen_golden 已处理）。
from __future__ import annotations

import torch


def copy_reference(dst: torch.Tensor, src: torch.Tensor,
                   non_blocking: bool = False) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        if isinstance(src, (int, float, bool)):
            return dst.fill_(src)
        raise TypeError("unsupported src type for copy_: ", type(src))
    return dst.copy_(src, non_blocking=non_blocking)
