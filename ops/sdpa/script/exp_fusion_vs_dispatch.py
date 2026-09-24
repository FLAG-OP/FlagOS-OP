# 实验A: torch.compile 子图融合能否改善单算子 SDPA 性能（预期:不能）
# 实验B: 混合路由（截流下发）原型——shape-aware dispatch 的收益
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    from kernel.triton_level import sdpa_triton

    F = torch.nn.functional
    g = torch.Generator(device="cpu").manual_seed(0)
    S, D, H = 2048, 128, 16
    q = torch.randn(1, H, S, D, generator=g).to(torch.float16).to("npu:0")
    k = torch.randn(1, H, S, D, generator=g).to(torch.float16).to("npu:0")
    v = torch.randn(1, H, S, D, generator=g).to(torch.float16).to("npu:0")

    print("== 实验A: torch.compile / graph 融合对单算子 SDPA 的作用 ==")
    t_eager_ours = bench(lambda: sdpa_triton(q, k, v, None, 0.0, True,
                                             None, False))
    t_eager_nat = bench(lambda: F.scaled_dot_product_attention(
        q, k, v, is_causal=True))
    print(f"eager ours : {t_eager_ours:.3f} ms")
    print(f"eager native: {t_eager_nat:.3f} ms")
    try:
        c_ours = torch.compile(sdpa_triton, dynamic=False)
        t = bench(lambda: c_ours(q, k, v, None, 0.0, True, None, False))
        print(f"compile ours: {t:.3f} ms  ({t/t_eager_ours:.2f}x eager)")
    except Exception as e:
        print(f"compile ours: 不支持 ({type(e).__name__}: "
              f"{str(e)[:100]})")
    try:
        c_nat = torch.compile(
            lambda q, k, v: F.scaled_dot_product_attention(
                q, k, v, is_causal=True), dynamic=False)
        t = bench(lambda: c_nat(q, k, v))
        print(f"compile native: {t:.3f} ms  ({t/t_eager_nat:.2f}x eager)")
    except Exception as e:
        print(f"compile native: 不支持 ({type(e).__name__}: "
              f"{str(e)[:100]})")

    print()
    print("== 实验B: 混合路由（截流下发）原型 ==")

    # shape-aware dispatch: 大 S（GEMM-bound, 栈差距大）→ 原生直通;
    # 小 S / native 不支持的组合 → 自研 Triton
    def sdpa_smart(q, k, v, attn_mask=None, is_causal=True,
                   enable_gqa=False):
        S = q.shape[-2]
        if S >= 1024 and attn_mask is None:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, enable_gqa=enable_gqa)
        return sdpa_triton(q, k, v, attn_mask, 0.0, is_causal, None,
                           enable_gqa)

    from script.perf_sweep import S_LIST  # noqa: F401  (仅示意, 下方手扫)
    import json
    rows = json.load(open(OP_DIR / "reports" / "perf_sweep.json"))
    print(f"{'S':>6} {'ours':>8} {'native':>8} {'smart':>8} {'smart/最优':>9}")
    print("-" * 44)
    for r in rows:
        if r["D"] != 128:
            continue
        S = r["S"]
        B = 1
        qq = torch.randn(B, H, S, D, generator=g).to(torch.float16).to("npu:0")
        kk = torch.randn(B, H, S, D, generator=g).to(torch.float16).to("npu:0")
        vv = torch.randn(B, H, S, D, generator=g).to(torch.float16).to("npu:0")
        t_s = bench(lambda: sdpa_smart(qq, kk, vv))
        best = min(r["ours_ms"], r["native_ms"])
        print(f"{S:>6} {r['ours_ms']:>8.3f} {r['native_ms']:>8.3f} "
              f"{t_s:>8.3f} {t_s/best:>8.2f}x")


if __name__ == "__main__":
    main()
