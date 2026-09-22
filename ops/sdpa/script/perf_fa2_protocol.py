# FA2 论文协议对标: A100 公开数据 vs Ascend910 实测
# 协议: batch = 16k / seqlen（总 token 恒定），D∈{64,128}，causal，fp16 前向
# A100 参照: FA2 论文 (arXiv:2307.08691) fig.4 前向 ~230 TFLOPS 峰值段
#            A100 80GB SXM fp16 峰值 312 TFLOPS
# 本机: Ascend910_9382, 24 AICore @1800MHz（峰值按官方 spec 表记）
import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

DEV = "npu:0"
DT = torch.float16
TOTAL_TOKENS = 16384
S_LIST = [512, 1024, 2048, 4096, 8192]
D_LIST = [64, 128]
H_LIST = {64: 32, 128: 16}          # FA2 协议: hidden 2048


def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def flops(S, D, H, B, causal=True):
    return 0.5 * 4 * B * H * S * S * D if causal else 4 * B * H * S * S * D


def main():
    from kernel.triton_level import sdpa_triton

    F = torch.nn.functional
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for D in D_LIST:
        H = H_LIST[D]
        for S in S_LIST:
            B = TOTAL_TOKENS // S
            q = torch.randn(B, H, S, D, generator=g).to(DT).to(DEV)
            k = torch.randn(B, H, S, D, generator=g).to(DT).to(DEV)
            v = torch.randn(B, H, S, D, generator=g).to(DT).to(DEV)
            fl = flops(S, D, H, B)

            t_ours = bench(lambda: sdpa_triton(q, k, v, None, 0.0, True,
                                               None, False))
            t_nat = bench(lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=True))
            rows.append({
                "S": S, "D": D, "H": H, "B": B,
                "ours_ms": round(t_ours, 4),
                "ours_tflops": round(fl / (t_ours / 1e3) / 1e12, 1),
                "native_ms": round(t_nat, 4),
                "native_tflops": round(fl / (t_nat / 1e3) / 1e12, 1),
            })
            print(f"D={D:3d} S={S:5d} B={B:3d}: ours={t_ours:8.3f}ms "
                  f"({rows[-1]['ours_tflops']:6.1f} TF)  "
                  f"nat={t_nat:7.3f}ms ({rows[-1]['native_tflops']:6.1f} TF)",
                  flush=True)
    out = Path(OP_DIR) / "reports" / "perf_fa2_protocol.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
