#!/usr/bin/env python3
"""样例 backward-example: 自定义算子的反向传播注册与验证。

此前样例全部为推理（forward-only）；本样例补齐训练侧:
  1. torch.autograd.Function 包装自定义 kernel（forward + backward）
  2. backward kernel 也用 Triton 实现
  3. 与 PyTorch autograd 数值梯度对比验证

运行: python3 examples/backward-example/example.py [设备profile名]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_and_mul_fwd(X, G, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    g = tl.load(G + cols, mask=mask, other=0.).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_in = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_in)
    tl.store(Y + cols, (gelu * g).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_and_mul_bwd(X, G, DY, DX, DG, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    g = tl.load(G + cols, mask=mask, other=0.).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_in = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    inner_p = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
    sech2 = 1.0 - tanh_in * tanh_in
    gelu = 0.5 * x * (1.0 + tanh_in)
    gelu_p = 0.5 * (1.0 + tanh_in) + 0.5 * x * sech2 * inner_p
    tl.store(DX + cols, (dy * gelu_p * g).to(DX.dtype.element_ty), mask=mask)
    tl.store(DG + cols, (dy * gelu).to(DG.dtype.element_ty), mask=mask)


class GeluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        from flag_gems.runtime import torch_device_fn
        x_flat = x.reshape(-1); g_flat = gate.reshape(-1)
        N = x_flat.numel(); out = torch.empty_like(x_flat)
        BLOCK = 1024; grid = (triton.cdiv(N, BLOCK),)
        with torch_device_fn.device(x.device):
            _gelu_and_mul_fwd[grid](x_flat, g_flat, out, N, BLOCK=BLOCK)
        ctx.save_for_backward(x_flat, g_flat)
        ctx.input_shapes = (x.shape, gate.shape)
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        from flag_gems.runtime import torch_device_fn
        x_flat, g_flat = ctx.saved_tensors
        dy = grad_out.reshape(-1); N = x_flat.numel()
        dx = torch.empty_like(x_flat); dg = torch.empty_like(g_flat)
        BLOCK = 1024; grid = (triton.cdiv(N, BLOCK),)
        with torch_device_fn.device(dy.device):
            _gelu_and_mul_bwd[grid](x_flat, g_flat, dy, dx, dg, N, BLOCK=BLOCK)
        return dx.reshape(ctx.input_shapes[0]), dg.reshape(ctx.input_shapes[1])


def run(profile) -> bool:
    dev = profile.torch_device
    print("=" * 60)
    print(f"backward-example: autograd fwd+bwd [{profile.name}] @ {dev}")
    print("=" * 60)

    # forward
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev) * 2
    g = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    out = GeluAndMulFunction.apply(x, g)
    xf, gf = x.float(), g.float()
    ref = (0.5 * xf * (1 + torch.tanh(0.7978845608028654 * (xf + 0.044715 * xf**3))) * gf).to(x.dtype)
    assert (out.float() - ref.float()).abs().max().item() < 1e-2
    print("  forward: PASS")

    # backward vs PyTorch autograd
    x_req = (torch.randn(32, 512, dtype=torch.float32, device=dev) * 2).requires_grad_(True)
    g_req = torch.randn(32, 512, dtype=torch.float32, device=dev).requires_grad_(True)
    out_c = GeluAndMulFunction.apply(x_req, g_req)
    go = torch.randn_like(out_c)
    dx_c, dg_c = torch.autograd.grad(out_c, (x_req, g_req), go)

    x_r = x_req.detach().requires_grad_(True)
    g_r = g_req.detach().requires_grad_(True)
    xf2, gf2 = x_r.float(), g_r.float()
    ref_o = (0.5 * xf2 * (1 + torch.tanh(0.7978845608028654 * (xf2 + 0.044715 * xf2**3))) * gf2).to(x_r.dtype)
    dx_r, dg_r = torch.autograd.grad(ref_o, (x_r, g_r), go)

    dx_err = (dx_c - dx_r).abs().max().item()
    dg_err = (dg_c - dg_r).abs().max().item()
    assert dx_err < 1e-3, f"dx err={dx_err}"
    assert dg_err < 1e-3, f"dg err={dg_err}"
    print(f"  backward: dx_err={dx_err:.1e} dg_err={dg_err:.1e} PASS")

    # 训练冒烟
    inp = torch.randn(16, 64, dtype=torch.float32, device=dev, requires_grad=True)
    gate = torch.randn(16, 64, dtype=torch.float32, device=dev, requires_grad=True)
    loss = GeluAndMulFunction.apply(inp, gate).sum()
    loss.backward()
    print("  训练冒烟: PASS")
    print("  => backward-example PASS")
    return True


# ============ 性能回归用例（scripts/perf_run.py 消费） ============
def perf_cases(profile):
    from common.perf import PerfCase

    def fwd(p):
        x = torch.randn(64, 4096, dtype=torch.bfloat16, device=p.torch_device) * 2
        g = torch.randn(64, 4096, dtype=torch.bfloat16, device=p.torch_device)
        return lambda: GeluAndMulFunction.apply(x, g)

    def bwd(p):
        x = (torch.randn(64, 4096, dtype=torch.bfloat16,
                         device=p.torch_device) * 2).requires_grad_(True)
        g = torch.randn(64, 4096, dtype=torch.bfloat16,
                        device=p.torch_device).requires_grad_(True)
        go = torch.randn(64, 4096, dtype=torch.bfloat16, device=p.torch_device)

        def step():
            if x.grad is not None:
                x.grad = None
                g.grad = None
            out = GeluAndMulFunction.apply(x, g)
            return torch.autograd.grad(out, (x, g), go)
        return step

    return [
        PerfCase("example.backward-example.gelu_and_mul.fwd", group="example",
                 level="kernel", make_fn=fwd),
        PerfCase("example.backward-example.gelu_and_mul.bwd", group="example",
                 level="kernel", make_fn=bwd),
    ]


if __name__ == "__main__":
    from common.device import load_profile
    run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin"))
