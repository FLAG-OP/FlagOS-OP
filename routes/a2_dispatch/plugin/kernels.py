# Triton kernel + PyTorch 参考实现
import json
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


# ============ gelu_and_mul（示例算子） ============
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gelu_and_mul_kernel(x, y):
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    return 0.5 * xf * (1.0 + tanh_inner) * y


def gelu_and_mul_triton(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return gelu_and_mul_kernel(x, gate)


def gelu_and_mul_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    gelu = 0.5 * xf * (1.0 + torch.tanh(inner))
    return (gelu * gate.float()).to(x.dtype)


# ============ silu_and_mul（框架级矩阵格用） ============
@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def silu_and_mul_kernel(x, y):
    xf = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-xf))
    return xf * sig * y


_SILU_COUNT_FILE = os.environ.get("A2_SILU_COUNT_FILE")


def silu_and_mul_triton_counted(x: torch.Tensor) -> torch.Tensor:
    """Triton silu_and_mul（vLLM 语义: silu(x1)*x2），带调用计数。

    框架级测试以 vendor:triton-template 身份注册并经 PER_OP 钉住。
    """
    d = x.shape[-1] // 2
    out = silu_and_mul_kernel(x[..., :d], x[..., d:])
    if _SILU_COUNT_FILE:
        counts = {}
        try:
            with open(_SILU_COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["silu_and_mul"] = counts.get("silu_and_mul", 0) + 1
        with open(_SILU_COUNT_FILE, "w") as f:
            json.dump(counts, f)
    return out
