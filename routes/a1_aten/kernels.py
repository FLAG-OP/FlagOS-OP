# Triton kernel: gelu (tanh approximation)
import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


# ============ 模板要点 ============
# 1. @pointwise_dynamic 自动处理 shape 广播、block 划分、多 dtype 分发
# 2. promotion_methods 声明输入到输出的类型提升规则
# 3. kernel 内部统一升 fp32 计算，保证 bf16/fp16 精度
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh_kernel(x):
    xf = x.to(tl.float32)
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    tanh_inner = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    return 0.5 * xf * (1.0 + tanh_inner)


def gelu_tanh_triton(x: torch.Tensor) -> torch.Tensor:
    return gelu_tanh_kernel(x)
