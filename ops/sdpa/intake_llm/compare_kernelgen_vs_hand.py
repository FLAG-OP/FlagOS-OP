# KernelGen 官方 MCP 生成版 vs 手写版 vs 本地 LLM-Track 对比
# 同一四阶段协议: 编译/正确性(36 case 同覆盖)/哨兵/性能
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from kernelgen_triton import scaled_dot_product_attention as sdpa_kg


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
    from kernel.triton_level import sdpa_triton
    from reference import sdpa_reference

    dev = "npu:0"
    dt = torch.float16
    g = torch.Generator(device="cpu").manual_seed(42)

    print("=" * 70)
    print("阶段①: 编译与首次运行（KernelGen 官方生成版）")
    print("=" * 70)
    q = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(1, 4, 128, 64, generator=g) * 0.5).to(dt).to(dev)
    try:
        out = sdpa_kg(q, k, v, None, 0.0, True)
        print("  编译+运行 OK", tuple(out.shape))
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return 1

    print()
    print("=" * 70)
    print("阶段②: 正确性（kernel 层同覆盖矩阵 + mask 路径）")
    print("=" * 70)
    cases = [
        ("basic-causal", 1, 4, 128, 64, True, None),
        ("batch", 2, 8, 256, 64, True, None),
        ("tail100", 1, 4, 100, 64, False, None),
        ("tail100-causal", 1, 4, 100, 64, True, None),
        ("d128", 2, 4, 512, 128, True, None),
        ("noncausal", 1, 4, 128, 64, False, None),
        ("boolmask", 1, 4, 128, 64, False, "bool"),
        ("floatmask", 1, 4, 128, 64, False, "float"),
        ("sq1-decode", 1, 4, 1, 64, False, None),
    ]
    n_pass = 0
    for tag, B, H, S, D, causal, mask in cases:
        qq = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        kk = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        vv = (torch.randn(B, H, S, D, generator=g) * 0.5).to(dt).to(dev)
        m = None
        if mask == "bool":
            m = (torch.rand(B, H, S, S, generator=g) > 0.3).to(dev)
        elif mask == "float":
            m = (torch.randn(B, H, S, S, generator=g) * 0.1).to(dt).to(dev)
        try:
            out = sdpa_kg(qq, kk, vv, m, 0.0, causal)
            ref = sdpa_reference(qq, kk, vv, m, 0.0, causal, None,
                                 False).float()
            err = (out.float() - ref).abs().max().item()
            ok = err < 2e-2
            n_pass += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag:14s} err={err:.3e}")
        except Exception as e:
            print(f"  [FAIL] {tag:14s} {type(e).__name__}: "
                  f"{str(e)[:90].replace(chr(10), ' ')}")
    print(f"  正确性: {n_pass}/{len(cases)}")

    print()
    print("=" * 70)
    print("阶段③: 哨兵（确定性 + 输入敏感; 二次调用抓 #11 静默 no-op）")
    print("=" * 70)
    o1 = sdpa_kg(q, k, v, None, 0.0, True)
    o2 = sdpa_kg(q, k, v, None, 0.0, True)
    det = torch.equal(o1, o2)
    o3 = sdpa_kg(q + 0.25, k, v, None, 0.0, True)
    sens = not torch.equal(o1, o3)
    print(f"  确定性: {'✓' if det else '✗'}  输入敏感: "
          f"{'✓' if sens else '✗'}")

    print()
    print("=" * 70)
    print("阶段④: 性能（S=2048 D=128 H=16 causal fp16）")
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
        t_kg = bench(lambda: sdpa_kg(q2, k2, v2, None, 0.0, True))
        print(f"  原生 CANN        : {t_nat:7.3f} ms")
        print(f"  手写版           : {t_hand:7.3f} ms  "
              f"({t_hand/t_nat:.1f}x native)")
        print(f"  KernelGen 官方版 : {t_kg:7.3f} ms  "
              f"({t_kg/t_nat:.1f}x native, {t_kg/t_hand:.2f}x 手写)")
    except Exception as e:
        print(f"  KernelGen 版性能不可测: {type(e).__name__}: "
              f"{str(e)[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
