# 性能根因探索: 为什么 ours 比原生 CANN 慢 6-21x
# 三组实验:
#   A. tile/warps/stages 网格扫描（cube 利用率假设）
#   B. LOAD_KT 变体（tl.trans 映射假设）
#   C. Triton 纯 matmul 天花板（栈上限 vs kernel 结构）
# 运行: python3 script/perf_explore.py > perf_explore.log
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def bench_fn(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def mk(B, H, S, D, dt, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
    k = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
    v = torch.randn(B, H, S, D, generator=g).to(dt).to(dev)
    return q, k, v


def flops(S, D, causal=True):
    # 每 head: QK^T + PV = 2 * S*S*D * 2 (causal 减半)
    f = 4 * S * S * D
    return f * (0.5 if causal else 1.0)


DEV = "npu:0"
DT = torch.float16

print("=" * 78)
print("A+B. SDPA tile/warps/stages/LOAD_KT 扫描 @ S=2048 D=128 H=16 causal")
print("=" * 78)
from kernel.triton_level import sdpa_triton  # noqa: E402

q, k, v = mk(1, 16, 2048, 128, DT, DEV)
t_native = bench_fn(lambda: torch.nn.functional
                    .scaled_dot_product_attention(q, k, v, is_causal=True))
print(f"native 基准: {t_native:.3f} ms  "
      f"({flops(2048, 128) * 16 / t_native / 1e9:.1f} TFLOPS)")
print(f"{'BM':>4s} {'BN':>4s} {'warps':>5s} {'stages':>6s} {'kt':>3s} "
      f"{'ms':>8s} {'TFLOPS':>7s} {'vs_nat':>7s}")
print("-" * 55)

results = []
for BM in (64, 128, 256):
    for BN in (64, 128):
        for W in (4, 8):
            for ST in (None, 2, 3):
                for KT in ((False,) if BM != 128 or W != 4 or ST is not None
                           else (False, True)):
                    try:
                        kw = dict(block_m=BM, block_n=BN, num_warps=W,
                                  load_kt=KT)
                        if ST is not None:
                            kw["num_stages"] = ST
                        t = bench_fn(lambda: sdpa_triton(
                            q, k, v, None, 0.0, True, None, False, **kw))
                        tf = flops(2048, 128) * 16 / t / 1e9
                        results.append((t, BM, BN, W, ST, KT, tf))
                        print(f"{BM:4d} {BN:4d} {W:5d} "
                              f"{str(ST):>6s} {int(KT):3d} {t:8.3f} "
                              f"{tf:7.1f} {t / t_native:6.2f}x")
                    except Exception as e:
                        print(f"{BM:4d} {BN:4d} {W:5d} {str(ST):>6s} "
                              f"{int(KT):3d}  FAIL {type(e).__name__} "
                              f"{str(e)[:40]}")

best = min(results)
print(f"\n最优: BM={best[1]} BN={best[2]} w={best[3]} st={best[4]} "
      f"kt={best[5]}  {best[0]:.3f} ms ({best[6]:.1f} TFLOPS, "
      f"{best[0] / t_native:.2f}x native)")

print()
print("=" * 78)
print("C. Triton 纯 matmul 天花板（同形状 GEMM，无 softmax）")
print("=" * 78)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _mm_kernel(A, B_, C, M, N, Kd,
               sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, Kd, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < Kd),
                    other=0.0)
        b = tl.load(B_ + offs_k[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=(offs_k[:, None] < Kd) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn,
             acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


from flag_gems.runtime import torch_device_fn  # noqa: E402

# SDPA S=2048 D=128 每 head 是两个 2048x128x2048 GEMM
M = N = 2048
Kd = 128
a = torch.randn(M, Kd, dtype=DT, device=DEV)
b_ = torch.randn(Kd, N, dtype=DT, device=DEV)
c = torch.empty(M, N, dtype=DT, device=DEV)
grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
with torch_device_fn.device(DEV):
    _mm_kernel[grid](a, b_, c, M, N, Kd,
                     a.stride(0), a.stride(1), b_.stride(0), b_.stride(1),
                     c.stride(0), c.stride(1), BM=128, BN=128, BK=64,
                     num_warps=8)
t_mm = bench_fn(lambda: _mm_kernel[grid](
    a, b_, c, M, N, Kd, a.stride(0), a.stride(1), b_.stride(0),
    b_.stride(1), c.stride(0), c.stride(1), BM=128, BN=128, BK=64,
    num_warps=8))
mm_flops = 2 * M * N * Kd
print(f"Triton matmul 2048x128x2048 fp16: {t_mm:.3f} ms "
      f"({mm_flops / t_mm / 1e9:.1f} GFLOPS/次)")

# 对照: torch.matmul 同形状
t_tmm = bench_fn(lambda: torch.matmul(a, b_))
print(f"torch.matmul 同形状:              {t_tmm:.3f} ms "
      f"({mm_flops / t_tmm / 1e9:.1f} GFLOPS)")

# SDPA 的两个 GEMM 每 head 共 2 次; 16 heads
est = t_mm * 2 * 16
print(f"\n推算: 16 heads × 2 GEMM = {est:.1f} ms —— SDPA 若 GEMM-bound "
      f"的 Triton 下限（不含 softmax 开销）")
print(f"native SDPA 实测 {t_native:.3f} ms —— 意味着原生单 GEMM 效率 "
      f"约为我们 Triton matmul 的 {est / (t_native * 1.0):.1f}x 倍")
