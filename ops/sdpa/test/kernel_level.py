# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics）。
# 运行: python3 test/kernel_level.py [ascend910|p800-kunlunxin]
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def _make_inputs(B, Hq, Hkv, Sq, Skv, D, dt, dev, seed, mask=None):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(B, Hq, Sq, D, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(dev)
    m = None
    if mask == "bool":
        m = (torch.rand(B, Hq, Sq, Skv_, generator=g) > 0.3).to(dev) \
            if False else (torch.rand(B, Hq, Sq, Skv, generator=g)
                           > 0.3).to(dev)
    elif mask == "float":
        m = (torch.randn(B, Hq, Sq, Skv, generator=g) * 0.1).to(dt).to(dev)
    return q, k, v, m


def run(profile):
    import torch

    from kernel.triton_level import sdpa_triton
    from reference import sdpa_reference

    dev = profile.torch_device
    results = []
    max_err_all = 0.0

    # ---------- 1) 精度: shape×dtype×mask 模式 vs fp32 参考 ----------
    # 覆盖: 尾块（S 非 BLOCK 整倍数）、长序列、GQA、D 非 2 幂、decode 形态
    cases = [
        # (B, Hq, Hkv, Sq, Skv, D, causal, mask, tag)
        (1, 4, 4, 128, 128, 64, False, None, "basic"),
        (1, 4, 4, 128, 128, 64, True, None, "causal"),
        (2, 8, 8, 256, 256, 64, True, None, "batch"),
        (1, 4, 4, 100, 100, 64, False, None, "tail100"),
        (1, 4, 4, 100, 100, 64, True, None, "tail100-causal"),
        (1, 4, 4, 197, 197, 80, True, None, "vit-d80"),
        (1, 8, 2, 256, 256, 64, True, None, "gqa"),
        (1, 8, 2, 256, 256, 64, False, "bool", "gqa-boolmask"),
        (1, 4, 4, 128, 128, 64, False, "float", "floatmask"),
        (1, 4, 4, 333, 333, 64, True, None, "tail333"),
        (2, 4, 4, 512, 512, 128, True, None, "d128"),
        (1, 4, 4, 1, 7, 64, False, None, "decode-sq1"),
    ]
    for (B_, Hq_, Hkv_, Sq_, Skv_, D_, causal, mask, tag) in cases:
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            q, k, v, m = _make_inputs(B_, Hq_, Hkv_, Sq_, Skv_, D_,
                                      dt, dev, 42, mask)
            out = sdpa_triton(q, k, v, m, 0.0, causal, None, Hq_ != Hkv_)
            # 黄金口径是 CPU fp32；P800 XMLIR 的组合 matmul 存在已知 JIT
            # 缺陷，不能把“判卷标准”混入被测设备栈。
            ref = sdpa_reference(
                q.cpu(), k.cpu(), v.cpu(),
                m.cpu() if m is not None else None,
                0.0, causal, None, Hq_ != Hkv_,
            ).to(dev).float()
            diff = (out.float() - ref).abs()
            # 全遮蔽行语义: 参考 NaN 行跳过（bool mask 可能产生）
            nan_rows = torch.isnan(ref[..., 0])           # (B,H,Sq)
            if nan_rows.any():
                keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
                diff = diff[keep]
            err = diff.max().item() if diff.numel() else 0.0
            tol = 1e-5 if dt == torch.float32 else 2e-2
            status = "OK" if err < tol else "FAIL"
            max_err_all = max(max_err_all, err)
            line = (f"  [{status}] {tag:14s} B{B_} H{Hq_}/{Hkv_} "
                    f"S{Sq_}/{Skv_} D{D_} "
                    f"{str(dt).split('.')[-1]:9s} causal={int(causal)} "
                    f"mask={mask or '-':5s} err={err:.3e}"
                    + ("  <== 超容差" if status == "FAIL" else ""))
            results.append(line)
            assert err < tol, line

    # ---------- 2) 哨兵: 确定性 + 输入敏感（#11 静默 no-op 类） ----------
    q, k, v, _ = _make_inputs(2, 4, 4, 256, 256, 64,
                              torch.float16, dev, 7)
    o1 = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    o2 = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    assert torch.equal(o1, o2), "同输入两次调用不一致（静默 no-op?）"
    o3 = sdpa_triton(q + 0.25, k, v, None, 0.0, True, None, False)
    assert not torch.equal(o1, o3), "输出不随输入变化"
    results.append("  [OK] 哨兵: 确定性 + 输入敏感")

    # ---------- 3) 性能: 短采样（正式对照走 script/bench_perf.py） ----------
    perf_ms = quick_perf(dev)

    for ln in results:
        print(ln)
    return {"ok": True, "max_err": round(max_err_all, 8),
            "cases": len(results),
            "sentinel": "deterministic+sensitive",
            "perf_ms_1k_ctx_d128": perf_ms}


def quick_perf(dev):
    import torch

    from kernel.triton_level import sdpa_triton
    q, k, v, _ = _make_inputs(1, 16, 16, 1024, 1024, 128,
                              torch.float16, dev, 0)
    def call():
        # XMLIR async guard: consume one output element, otherwise this can
        # measure only kernel submission rather than execution.
        return sdpa_triton(q, k, v, None, 0.0, True, None,
                           False)[0, 0, 0, 0].item()

    for _ in range(10):
        call()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(50):
        call()
    _sync(dev)
    return round((time.perf_counter() - t0) / 50 * 1000, 4)


def _sync(dev):
    import torch
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev.startswith("mlu"):
        torch.mlu.synchronize()


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else None)))
