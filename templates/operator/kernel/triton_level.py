# Triton 级实现（占位名 my_op，使用时全局替换）。
#
# 硬约束（违反必错，均来自本库实测）:
#   1. 启动必须包 torch_device_fn.device 上下文（known-issues #11）
#   2. 尾块（N 非 BLOCK 整数倍）masked load + tl.sum 会被污染，other/
#      tl.where 均无法补救——pad 到整倍数或两阶段归约（known-issues #15a）
#   3. 不用 @triton.autotune（本栈会选出非法 num_warps=5，#15b）
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 1024                    # 固定配置（勿用 autotune，#15b）


@triton.jit
def _my_op_kernel(X, G, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float32)     # 输入已 pad，无需 mask（#15a）
    g = tl.load(G + offs).to(tl.float32)
    y = x * g                                # <TODO: fp32 内部计算>
    tl.store(Y + offs, y.to(Y.dtype.element_ty))


def my_op_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """<TODO: 一句话语义与签名说明>。"""
    from flag_gems.runtime import torch_device_fn

    shape = x.shape
    n = x.numel()
    pad = (-n) % BLOCK                        # 尾块安全（#15a）
    if pad:
        x = torch.nn.functional.pad(x.reshape(-1), (0, pad))
        gate = torch.nn.functional.pad(gate.reshape(-1), (0, pad))
    out = torch.empty_like(x.reshape(-1))
    with torch_device_fn.device(x.device):    # ← 硬约束 #11
        _my_op_kernel[(triton.cdiv(n + pad, BLOCK),)](
            x.reshape(-1), gate.reshape(-1), out, n + pad, BLOCK=BLOCK)
    return out[:n].reshape(shape)
