# 多尺度性能扫描: S × D 网格（三方实现），产出 sweep JSON + 三张图
# 运行: python3 script/perf_sweep.py   (~20-30 分钟，含编译)
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

S_LIST = [256, 512, 1024, 2048, 4096, 8192]     # 序列长度 6 点
D_LIST = [64, 128, 256]                          # head_dim 3 点
H = 16
DT = torch.float16
DEV = "npu:0"


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
    rows = []
    g = torch.Generator(device="cpu").manual_seed(0)

    for D in D_LIST:
        for S in S_LIST:
            q = torch.randn(1, H, S, D, generator=g).to(DT).to(DEV)
            k = torch.randn(1, H, S, D, generator=g).to(DT).to(DEV)
            v = torch.randn(1, H, S, D, generator=g).to(DT).to(DEV)
            t_nat = bench(lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=True))
            t_ours = bench(lambda: sdpa_triton(q, k, v, None, 0.0, True,
                                               None, False))
            try:
                from flag_gems.ops.attention import (
                    scaled_dot_product_attention_forward)
                t_gems = bench(lambda: scaled_dot_product_attention_forward(
                    q, k, v, None, 0.0, True, None, False))
            except Exception:
                t_gems = None
            rows.append({"S": S, "D": D, "H": H,
                         "ours_ms": round(t_ours, 4),
                         "native_ms": round(t_nat, 4),
                         "gems_ms": round(t_gems, 4) if t_gems else None})
            print(f"D={D:3d} S={S:5d}: ours={t_ours:8.3f} "
                  f"nat={t_nat:7.3f} gems={t_gems if t_gems else float('nan'):8.3f}"
                  f"  ratio={t_ours/t_nat:5.2f}x", flush=True)

    out = Path(OP_DIR) / "reports" / "perf_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
