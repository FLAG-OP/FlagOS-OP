# [KernelGen intake] 生成产物演示（正例）
# 实际使用时此文件由 KernelGen / KernelBench / Agent 产出，此处以手写
# Triton kernel 演示契约形状。
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _gelu_and_mul_kernel(X, G, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_in = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    y = 0.5 * x * (1.0 + tanh_in) * g
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def gelu_and_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    # 厂商栈要求: Triton 启动包 device 上下文（known-issues #11）
    with torch_device_fn.device(x.device):
        _gelu_and_mul_kernel[grid](
            x.reshape(-1), gate.reshape(-1), out.reshape(-1), n,
            BLOCK=1024)
    return out
