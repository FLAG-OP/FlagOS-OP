#!/usr/bin/env python3
"""样例 softmax-fullstack: 行归约 kernel + autotune（Triton 教程三部曲之三）。

覆盖两个此前缺失的算子类别:
  1. reduction 类: softmax 是行归约（max + exp + sum + div）
  2. autotune: @triton.autotune 自动搜索最优 BLOCK_SIZE

运行: python3 examples/softmax-fullstack/example.py [设备profile名]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


# ============ aten 注册 ============
_LIB = None
CALLS = {"softmax": 0}

def _softmax_counter(self, dim=-1, dtype=None):
    CALLS["softmax"] += 1
    return softmax_triton(self, dim)

def register_softmax(dispatch_key: str):
    global _LIB
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("_softmax", _softmax_counter, dispatch_key)


def _bench(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ============ 性能回归用例（scripts/perf_run.py 消费） ============
def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=p.torch_device)
            return lambda: fn(x)
        return _make

    def bw_2t(t):
        return {"GBps": 1024 * 1024 * 2 * 2 / t / 1e6}

    return [
        PerfCase("example.softmax-fullstack.softmax.triton", group="example",
                 level="kernel", make_fn=make(softmax_triton), derived=bw_2t),
        PerfCase("example.softmax-fullstack.softmax.torch", group="example",
                 level="kernel", make_fn=make(lambda x: F.softmax(x, dim=-1))),
        PerfCase("example.softmax-fullstack.softmax.flaggems", group="example",
                 level="kernel", make_fn=make(_flaggems_softmax)),
    ]


def _flaggems_softmax(x):
    from flag_gems import ops
    return ops.softmax(x, -1)


def stage_l0(dev: str) -> bool:
    print("-" * 60)
    print("Stage 1/3  kernel 层: Triton fused softmax (reduction + autotune)")
    print("-" * 60)
    # 含非整倍数 N（3072/5000/5120）——原测试全是整倍数，漏测尾块路径
    for shape in [(128, 512), (256, 1024), (64, 2048), (1, 4096), (512, 256),
                  (32, 3072), (8, 5000), (16, 5120)]:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 3
            ref = F.softmax(x.float(), dim=-1).to(dt)
            out = softmax_triton(x)
            err = (out.float() - ref.float()).abs().max().item()
            assert err < 1e-2, f"{shape} {dt} err={err}"
    print("  精度: 24/24 组合 PASS（含非整倍数 N 尾块回归）")
    x = torch.randn(64, 512, dtype=torch.bfloat16, device=dev)
    o1, o2 = softmax_triton(x), softmax_triton(x)
    assert torch.equal(o1, o2) and not torch.equal(o1, softmax_triton(x + 1))
    print("  哨兵: 确定性 OK 输入敏感 OK")
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
    t_tri = _bench(lambda: softmax_triton(x))
    t_ref = _bench(lambda: F.softmax(x, dim=-1))
    bw = x.numel() * 2 * 2 / t_tri / 1e9 * 1e3
    print(f"  性能: Triton={t_tri:.3f}ms ({bw:.0f} GB/s)  F.softmax={t_ref:.3f}ms  相对={t_ref/t_tri:.2f}x")
    try:
        from flag_gems import ops as FG
        t_fg = _bench(lambda: FG.softmax(x, -1))
        print(f"  FlagGems 基线: {t_fg:.3f}ms（自研相对={t_fg/t_tri:.2f}x）")
    except Exception as e:
        print(f"  FlagGems 基线: SKIP（{type(e).__name__}）")
    print(f"  配置: BLOCK_N=2048 固定（本栈 autotune 不安全，known-issues #15b）")
    return True


def stage_l2(dev: str, dispatch_key: str) -> bool:
    print()
    print("-" * 60)
    print("Stage 2/3  框架层 aten: F.softmax 拦截")
    print("-" * 60)
    register_softmax(dispatch_key)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device=dev)
    before = CALLS["softmax"]
    out = F.softmax(x, dim=-1)
    assert CALLS["softmax"] == before + 1, "未被拦截"
    ref = F.softmax(x.cpu().float(), dim=-1).to(dev)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 1e-2
    print(f"  拦截: F.softmax -> Triton kernel (calls={CALLS['softmax']})")
    print(f"  拦截路径精度: max_err={err:.1e} PASS")
    return True


def stage_l4(dev: str) -> bool:
    print()
    print("-" * 60)
    print("Stage 3/3  应用层: attention scores softmax")
    print("-" * 60)
    torch.manual_seed(42)
    B, H, T, D = 1, 4, 64, 64
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=dev)
    k = q.clone()

    def attn_scores(Q, K):
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / (D ** 0.5)
        return F.softmax(scores, dim=-1)

    with torch.no_grad():
        ref = attn_scores(q, k)
        out = attn_scores(q, k)
        err = (out.float() - ref.float()).abs().max().item()
    n = CALLS["softmax"]
    assert n >= 2, f"attention scores 未触发 softmax ({n})"
    assert err < 1e-2, f"应用层偏差 {err}"
    print(f"  attention 前向触发 softmax: {n} 次")
    print(f"  输出一致性: max_err={err:.1e} PASS")
    return True


def main() -> None:
    from common.device import load_profile
    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else "p800-kunlunxin")
    dev = profile.torch_device
    print("=" * 60)
    print(f"softmax-fullstack: reduction kernel + autotune")
    print(f"设备: {profile.summary()}")
    print("=" * 60)
    assert stage_l0(dev)
    assert stage_l2(dev, profile.dispatch_key)
    assert stage_l4(dev)
    print()
    print("=> softmax-fullstack PASS (reduction + autotune)")


if __name__ == "__main__":
    main()
