#!/usr/bin/env python3
"""T1 任务书 §2b: Triton-on-PPU 最小 tl.dot 冒烟（编译+启动+对点）。

在空卡（默认 cuda:6）上跑一个 64x64 fp32 全幅 tl.dot matmul，与
torch matmul 比对，JSON 结果落 probes/out/triton_dot_smoke.json。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl

OUT = Path(__file__).resolve().parent / "out" / "triton_dot_smoke.json"
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:6"
BLOCK = 64


@triton.jit
def dot_kernel(A, B, C, stride_am, stride_ak, stride_bk, stride_bn,
               stride_cm, stride_cn, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    a = tl.load(A + offs[:, None] * stride_am + offs[None, :] * stride_ak)
    b = tl.load(B + offs[:, None] * stride_bk + offs[None, :] * stride_bn)
    c = tl.dot(a, b)
    tl.store(C + offs[:, None] * stride_cm + offs[None, :] * stride_cn, c)


def main() -> int:
    torch.cuda.set_device(torch.device(DEV))
    a = torch.randn(BLOCK, BLOCK, device=DEV)
    b = torch.randn(BLOCK, BLOCK, device=DEV)
    c = torch.empty(BLOCK, BLOCK, device=DEV)
    try:
        dot_kernel[(1, 1)](a, b, c, BLOCK, 1, BLOCK, 1, BLOCK, 1, BLOCK=BLOCK)
        torch.cuda.synchronize()
        ref = a @ b
        err = (c - ref).abs().max().item()
        scale = ref.abs().max().item()
        rel = err / max(scale, 1e-12)
        # PPU 实测: fp32 输入下 tl.dot 精度掉到 fp16 量级（abs 2.4e-2 @
        # |c|max≈20, rel≈1.2e-3），与 ieee-sdpa triton 1e-4 级偏差同因
        # ——对点判 rel。
        ok = rel < 5e-3
        meta = {"device": DEV, "gpu_name": torch.cuda.get_device_name(),
                "triton": triton.__version__, "torch": torch.__version__,
                "err_abs": err, "ref_scale": scale, "err_rel": rel,
                "note": "fp32 输入 tl.dot 实测精度≈fp16 量级(rel~1e-3)，"
                        "故 fp32 路径不走 triton", "ok": ok}
        print(f"[triton-dot-smoke] {meta}")
        status = 0 if ok else 1
    except Exception as exc:  # 编译/启动失败 -> 停点证据
        meta = {"device": DEV, "error": f"{type(exc).__name__}: {exc}",
                "ok": False}
        print(f"[triton-dot-smoke] FAIL {meta}")
        status = 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta["timestamp"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"[triton-dot-smoke] -> {OUT}")
    return status


if __name__ == "__main__":
    sys.exit(main())
