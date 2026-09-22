# LLM-Track 生成版 vs 手写版 对比实验（KernelGenBench 方法论）
# 阶段: ①编译/运行 ②正确性（黄金子集 + kernel 层 36 case）
#       ③哨兵（FlagOS-OP intake 协议——抓静默 no-op）④性能（同基准）
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def bench(fn, warmup=15, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    from intake_llm.kernel_llm_track import sdpa_llm_generated
    from kernel.triton_level import sdpa_triton
    from reference import sdpa_reference

    dev = "npu:0"
    dt = torch.float16
    g = torch.Generator(device="cpu").manual_seed(42)

    print("=" * 70)
    print("阶段①: 编译与首次运行")
    print("=" * 70)
    q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    try:
        out = sdpa_llm_generated(q, k, v, None, 0.0, True)
        print("  编译+运行 OK", tuple(out.shape))
    except Exception as e:
        print(f"  编译/运行 FAIL: {type(e).__name__}: {str(e)[:150]}")
        print("  → LLM-Track 终止于此（KernelGenBench 记为 pass=0）")
        return 1

    print()
    print("=" * 70)
    print("阶段②: 正确性（kernel 层同覆盖矩阵，生成版能力范围内的子集）")
    print("=" * 70)
    cases = [
        ("basic-causal", 1, 4, 128, 64, True),
        ("batch", 2, 8, 256, 64, True),
        ("tail100", 1, 4, 100, 64, False),   # 尾块 + 无 mask 路径
        ("d128", 2, 4, 512, 128, True),
        ("noncausal", 1, 4, 128, 64, False),
    ]
    n_pass = 0
    for tag, B, H, S, D, causal in cases:
        qq = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        kk = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        vv = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        try:
            out = sdpa_llm_generated(qq, kk, vv, None, 0.0, causal)
            ref = sdpa_reference(qq, kk, vv, None, 0.0, causal, None,
                                 False).float()
            err = (out.float() - ref).abs().max().item()
            ok = err < 2e-2
            n_pass += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag:12s} err={err:.3e}")
        except Exception as e:
            print(f"  [FAIL] {tag:12s} {type(e).__name__}: "
                  f"{str(e)[:80].replace(chr(10), ' ')}")
    print(f"  正确性: {n_pass}/{len(cases)}")

    print()
    print("=" * 70)
    print("阶段③: 哨兵（FlagOS-OP intake 协议: 确定性 + 输入敏感）")
    print("=" * 70)
    try:
        o1 = sdpa_llm_generated(q, k, v, None, 0.0, True)
        o2 = sdpa_llm_generated(q, k, v, None, 0.0, True)
        det = torch.equal(o1, o2)
        o3 = sdpa_llm_generated(q + 0.25, k, v, None, 0.0, True)
        sens = not torch.equal(o1, o3)
        print(f"  确定性: {'✓' if det else '✗'}  输入敏感: "
              f"{'✓' if sens else '✗'}")
    except Exception as e:
        print(f"  哨兵 FAIL: {type(e).__name__}")

    print()
    print("=" * 70)
    print("阶段④: 性能（同基准三方对照口径）")
    print("=" * 70)
    S, D, H = 2048, 128, 16
    q2 = (torch.randn(1, H, S, D, generator=g)).to(dt).to(dev)
    k2 = (torch.randn(1, H, S, D, generator=g)).to(dt).to(dev)
    v2 = (torch.randn(1, H, S, D, generator=g)).to(dt).to(dev)
    F = torch.nn.functional
    t_nat = bench(lambda: F.scaled_dot_product_attention(
        q2, k2, v2, is_causal=True))
    t_hand = bench(lambda: sdpa_triton(q2, k2, v2, None, 0.0, True, None,
                                       False))
    try:
        t_llm = bench(lambda: sdpa_llm_generated(q2, k2, v2, None, 0.0, True))
        print(f"  原生 CANN : {t_nat:7.3f} ms")
        print(f"  手写版    : {t_hand:7.3f} ms  ({t_hand/t_nat:.1f}x native)")
        print(f"  LLM 生成版: {t_llm:7.3f} ms  ({t_llm/t_nat:.1f}x native, "
              f"{t_llm/t_hand:.2f}x 手写)")
    except Exception as e:
        print(f"  LLM 版性能不可测: {type(e).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
