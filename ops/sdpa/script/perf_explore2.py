# 补充实验: ① BM=128/BN=32 能否绕过 UB 限制 ② fp32 性能 ③ 带宽/利用率核算
import sys
import time

sys.path.insert(0, "/root/sdpatten-op")
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernel.triton_level import sdpa_triton  # noqa: E402


def bench_fn(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


g = torch.Generator(device="cpu").manual_seed(0)
DEV = "npu:0"

for dt_name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16),
                    ("fp32", torch.float32)]:
    q = torch.randn(1, 16, 2048, 128, generator=g).to(dt).to(DEV)
    k = torch.randn(1, 16, 2048, 128, generator=g).to(dt).to(DEV)
    v = torch.randn(1, 16, 2048, 128, generator=g).to(dt).to(DEV)
    t_nat = bench_fn(lambda: torch.nn.functional
                     .scaled_dot_product_attention(q, k, v, is_causal=True))
    t_64 = bench_fn(lambda: sdpa_triton(q, k, v, None, 0.0, True, None,
                                        False))
    print(f"{dt_name}: native={t_nat:.3f}ms  ours(64x64)={t_64:.3f}ms "
          f"({t_64/t_nat:.1f}x)")

print("\nUB 绕过尝试 (fp16, S=2048 D=128):")
q = torch.randn(1, 16, 2048, 128, generator=g).to(torch.float16).to(DEV)
k = torch.randn(1, 16, 2048, 128, generator=g).to(torch.float16).to(DEV)
v = torch.randn(1, 16, 2048, 128, generator=g).to(torch.float16).to(DEV)
for bm, bn in [(128, 32), (128, 64), (256, 32), (64, 32)]:
    try:
        t = bench_fn(lambda: sdpa_triton(q, k, v, None, 0.0, True, None,
                                         False, block_m=bm, block_n=bn))
        print(f"  BM={bm} BN={bn}: {t:.3f} ms  OK")
    except Exception as e:
        print(f"  BM={bm} BN={bn}: FAIL {type(e).__name__} "
              f"{str(e)[:60].replace(chr(10), ' ')}")

# 精度抽查: BM=128 变体（若可编译）数值不变
print("\n带宽容算 @ S=2048 D=128 H=16 causal fp16:")
S, D, H = 2048, 128, 16
fl = 0.5 * 4 * S * S * D * H / 1e9       # GFLOP
kv_bytes = S * D * 2 * 2 * H             # K+V 全量 fp16
grid_m = S // 64
print(f"  FLOPs={fl:.1f}G  K/V={kv_bytes/1e6:.0f}MB  grid_m={grid_m}")
print(f"  K/V 名义流量(×grid_m)={kv_bytes*grid_m/1e6:.0f}MB")
t_ours = bench_fn(lambda: sdpa_triton(q, k, v, None, 0.0, True, None, False))
print(f"  ours {t_ours:.3f}ms → {fl/t_ours:.1f} TFLOPS, "
      f"名义带宽需 {kv_bytes*grid_m/t_ours/1e6:.0f} GB/s（HBM 峰值~1200）")
t_nat = bench_fn(lambda: torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True))
print(f"  native {t_nat:.3f}ms → {fl/t_nat:.1f} TFLOPS, "
      f"若 L2 全命中 K/V 仅需 {kv_bytes/t_nat/1e6:.0f} GB/s HBM")
