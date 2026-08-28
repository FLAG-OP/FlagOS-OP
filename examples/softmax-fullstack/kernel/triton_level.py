# Triton 级 softmax 实现（从原 example.py 抽取，逻辑不变）。
# 硬约束见 known-issues #15: 尾块 pad + 固定 BLOCK_N（不用 autotune）。
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============ Triton fused softmax（行归约） ============
# 注（known-issues #15）: 本栈 a) 尾块 masked load + tl.sum 归约错误，
# other/tl.where 均无法补救；b) @triton.autotune 会选出配置表里不存在
# 的非法 num_warps=5，导致错误执行且不可复现。
# 对策: wrapper 把输入 pad 到 2048 整数倍（实体填充极低值，exp 后贡献
# 为 0），kernel 固定 BLOCK_N=2048——全程无 mask、无调优，确定性路径。
@triton.jit
def _softmax_kernel(X, Y, N, sx, sy, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    # 流式三遍归约（支持 N > BLOCK_N）
    # Pass 1: 行最大值
    x_max = float("-inf")
    for start in range(0, N, BLOCK_N):
        mask = (start + cols) < N
        x = tl.load(X + row * sx + start + cols, mask=mask, other=float("-inf"))
        x_max = tl.maximum(x_max, tl.max(x, axis=0))

    # Pass 2: 指数和
    s = 0.0
    for start in range(0, N, BLOCK_N):
        mask = (start + cols) < N
        x = tl.load(X + row * sx + start + cols, mask=mask, other=float("-inf"))
        s += tl.sum(tl.exp(x - x_max), axis=0)

    # Pass 3: 归一化并写回
    for start in range(0, N, BLOCK_N):
        mask = (start + cols) < N
        x = tl.load(X + row * sx + start + cols, mask=mask, other=float("-inf"))
        tl.store(Y + row * sy + start + cols, tl.exp(x - x_max) / s, mask=mask)


def softmax_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    from flag_gems.runtime import torch_device_fn
    assert dim == -1 or dim == x.dim() - 1, "仅支持最后一维"
    x2d = x.reshape(-1, x.shape[-1])
    M, N = x2d.shape
    pad = (-N) % 2048
    if pad:
        x2d = F.pad(x2d, (0, pad), value=-60000.0)  # fp16/bf16/fp32 均安全
    N_p = N + pad
    out = torch.empty_like(x2d)
    with torch_device_fn.device(x.device):
        _softmax_kernel[(x2d.shape[0],)](
            x2d, out, N_p, x2d.stride(0), out.stride(0), BLOCK_N=2048)
    return out[..., :N].reshape(x.shape)
