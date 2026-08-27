# [KernelGen intake] 负例: 复现 known-issues #11（P800 裸 Triton 静默 no-op）
from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    # 缺 torch_device_fn.device 上下文 → P800 上后续启动静默 no-op
    _gelu_and_mul_kernel[grid](
        x.reshape(-1), gate.reshape(-1), out.reshape(-1), n, BLOCK=1024)
    return out
