# kernel 层直测: 精度 / 哨兵 / dropout / 错误路径 / 短性能（返回 metrics）。
# 运行: python3 test/kernel_level.py [ascend910|cpu]
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def _make(B, Hq, Hkv, Sq, Skv, D, dt, dev, seed, mask=None,
          gqa=False, causal=False, dropout_p=0.0):
    import torch
    from reference import make_dropout_mask
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(B, Hq, Sq, D, generator=g) * 0.5).to(dt).to(dev)
    k = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(dev)
    v = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).to(dt).to(dev)
    m = None
    if mask == "bool":
        m = (torch.rand(B, Hq, Sq, Skv, generator=g) > 0.3).to(dev)
    elif mask == "float":
        m = (torch.randn(B, Hq, Sq, Skv, generator=g) * 0.1).to(dt).to(dev)
    elif mask == "bool2d":
        m = (torch.rand(Sq, Skv, generator=g) > 0.3).to(dev)
    elif mask == "bool_b1":
        m = (torch.rand(B, 1, Sq, Skv, generator=g) > 0.3).to(dev)
    dm = None
    if dropout_p > 0:
        dm = make_dropout_mask((B, Hq, Sq, Skv), dropout_p, dev)
    return q, k, v, m, dm


def _impl_for(profile):
    """按 profile 选被测实现（ascend910 → triton；cpu → torch）。"""
    if profile.vendor == "ascend":
        from kernel.triton_level import sdpa_math_triton as fn
    else:
        from kernel.torch_level import sdpa_math_torch as fn
    return fn


def run(profile):
    import torch
    from reference import sdpa_math_reference

    dev = profile.torch_device
    fn = _impl_for(profile)
    results = []
    max_err = max_perr = 0.0

    # ── 1) 精度: shape × dtype × mask 模式 vs CPU 参考（out 与 P 双判卷）──
    cases = [
        # (B,Hq,Hkv,Sq,Skv,D, causal, mask, gqa, dropout_p, tag)
        (1, 4, 4, 128, 128, 64, False, None, False, 0.0, "basic"),
        (1, 4, 4, 128, 128, 64, True, None, False, 0.0, "causal"),
        (2, 8, 8, 256, 256, 64, True, None, False, 0.0, "batch-causal"),
        (1, 4, 4, 100, 100, 64, False, None, False, 0.0, "tail100"),
        (1, 4, 4, 197, 197, 80, True, None, False, 0.0, "vit-d80"),
        (1, 8, 2, 256, 256, 64, True, None, True, 0.0, "gqa"),
        (1, 8, 2, 256, 256, 64, False, "bool", True, 0.0, "gqa-boolmask"),
        (1, 4, 4, 128, 128, 64, False, "float", False, 0.0, "floatmask"),
        (1, 4, 4, 100, 100, 64, False, "bool2d", False, 0.0, "mask2d"),
        (2, 4, 4, 64, 64, 64, False, "bool_b1", False, 0.0, "mask-b1h"),
        (1, 4, 4, 1, 7, 64, False, None, False, 0.0, "decode-sq1"),
        (1, 4, 4, 1, 64, 64, True, None, False, 0.0, "decode-causal"),
        (1, 4, 4, 6, 4, 64, True, None, False, 0.0, "nonsq-sq>skv"),
        (1, 4, 4, 4, 6, 64, True, None, False, 0.0, "nonsq-sq<skv"),
        (1, 4, 4, 64, 64, 64, True, None, False, 0.3, "dropout-causal"),
        (1, 4, 4, 64, 64, 64, False, None, False, 0.5, "dropout-50"),
    ]
    for (B, Hq, Hkv, Sq, Skv, D, causal, mask, gqa, dp, tag) in cases:
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            q, k, v, m, dm = _make(B, Hq, Hkv, Sq, Skv, D, dt, dev, 42,
                                   mask, gqa, causal, dp)
            out, probs = fn(q, k, v, m, dp, causal, dm, enable_gqa=gqa)
            ref = sdpa_math_reference(
                q.cpu(), k.cpu(), v.cpu(),
                m.cpu() if m is not None else None, dp, causal,
                dm.cpu() if dm is not None else None, enable_gqa=gqa)
            eo = (out.float().cpu() - ref[0].float()).abs().max().item()
            ep = (probs.float().cpu() - ref[1].float()).abs().max().item()
            tol = 1e-5 if dt == torch.float32 else 2e-2
            ok = max(eo, ep) < tol
            max_err, max_perr = max(max_err, eo), max(max_perr, ep)
            line = (f"  [{'OK' if ok else 'FAIL'}] {tag:15s} "
                    f"B{B} H{Hq}/{Hkv} S{Sq}x{Skv} D{D} "
                    f"{str(dt).split('.')[-1]:9s} causal={int(causal)} "
                    f"mask={mask or '-':8s} gqa={int(gqa)} p={dp} "
                    f"out={eo:.2e} P={ep:.2e}")
            results.append(line)
            assert ok, line

    # ── 2) 全遮蔽行: P=0、out=0（不是 NaN），且概率行和正确 ──
    q, k, v, _, _ = _make(1, 2, 2, 8, 8, 16, torch.float32, dev, 5)
    bad = torch.zeros(1, 2, 8, 8, dtype=torch.bool, device=dev)  # 全 False
    out, probs = fn(q, k, v, bad, 0.0, False, None)
    assert torch.isfinite(out).all() and torch.isfinite(probs).all(), \
        "全遮蔽行出现 NaN/Inf"
    assert out.abs().max().item() == 0.0 and probs.abs().max().item() == 0.0, \
        "全遮蔽行应为 0（native 语义）"
    results.append("  [OK] 全遮蔽行 → out=P=0（无 NaN）")

    # ── 3) 哨兵: 确定性 + 输入敏感（#11 静默 no-op 类）──
    q, k, v, m, _ = _make(2, 4, 4, 256, 256, 64, torch.float16, dev, 7,
                          "bool")
    o1 = fn(q, k, v, m, 0.0, False, None)   # bool mask 与 causal 互斥
    o2 = fn(q, k, v, m, 0.0, False, None)
    assert torch.equal(o1[0], o2[0]) and torch.equal(o1[1], o2[1]), \
        "同输入两次调用不一致（静默 no-op?）"
    o3 = fn(q + 0.25, k, v, m, 0.0, False, None)
    assert not torch.equal(o1[0], o3[0]), "输出不随输入变化"
    results.append("  [OK] 哨兵: 确定性 + 输入敏感")

    # ── 4) dropout: 显式掩码逐位可判卷；随机路径统计性质 ──
    q, k, v, m, dm = _make(2, 4, 4, 128, 128, 64, torch.float32, dev, 9,
                           dropout_p=0.5)
    out, probs = fn(q, k, v, None, 0.5, False, dm)
    ref = sdpa_math_reference(q.cpu(), k.cpu(), v.cpu(), None, 0.5, False,
                              dm.cpu())
    eo = (out.cpu() - ref[0]).abs().max().item()
    ep = (probs.cpu() - ref[1]).abs().max().item()
    assert max(eo, ep) < 1e-5, f"显式 dropout 掩码不一致: {eo}/{ep}"
    # native 怪癖自检: 显式掩码下 O = (P/(1-p))@V（不是 P@V）
    vo = (probs.cpu().float() / 0.5) @ v.cpu().float()
    assert (out.cpu().float() - vo).abs().max().item() < 1e-4, \
        "显式 dropout 下 O ≠ (P/(1-p))@V"
    # 随机路径: keep 比例 ≈ 1-p，kept 上 P 相对 pristine ≈ 1/(1-p)
    _, p_clean = fn(q, k, v, None, 0.0, False, None)
    _, p_rand = fn(q, k, v, None, 0.5, False, None)
    zero = (p_rand == 0)
    keep_rate = 1.0 - zero.float().mean().item()
    ratio = (p_rand[~zero] / p_clean[~zero])
    r_med = ratio.median().item() if ratio.numel() else 0.0
    assert 0.42 < keep_rate < 0.58, f"随机 keep 比例异常: {keep_rate}"
    assert abs(r_med - 2.0) < 0.15, f"kept 上缩放因子异常: {r_med}"
    results.append(f"  [OK] dropout: 显式掩码一致 + 随机 keep={keep_rate:.3f} "
                   f"×{r_med:.3f}")

    # ── 5) 错误路径: causal+mask 冲突、GQA 头不整除 ──
    q, k, v, m, _ = _make(1, 4, 4, 32, 32, 64, torch.float32, dev, 3, "bool")
    try:
        fn(q, k, v, m, 0.0, True, None)
        raise AssertionError("causal+attn_mask 未报冲突")
    except RuntimeError as e:
        assert "Explicit attn_mask should not be set" in str(e)
    q8 = torch.randn(1, 7, 32, 16, device=dev)
    k2 = torch.randn(1, 2, 32, 16, device=dev)
    try:
        fn(q8, k2, k2, None, 0.0, False, None, enable_gqa=True)
        raise AssertionError("GQA 头不整除未报错")
    except RuntimeError as e:
        assert "divide" in str(e)
    results.append("  [OK] 错误路径: causal+mask 冲突 / GQA 不整除")

    # ── 6) 短性能（正式对照走 script/bench_perf.py）──
    perf_ms = quick_perf(dev, fn)

    for ln in results:
        print(ln)
    return {"ok": True, "impl": fn.__name__,
            "max_err": round(max_err, 8), "max_probs_err": round(max_perr, 8),
            "cases": len(results),
            "sentinel": "deterministic+sensitive",
            "dropout": "explicit+random-stats",
            "perf_ms_1k_ctx_d128": perf_ms}


def quick_perf(dev, fn):
    import torch
    q, k, v, _, _ = _make(1, 16, 16, 1024, 1024, 128, torch.float16, dev, 0)

    def call():
        # 读一个输出元素: 防异步栈只测到提交（XMLIR/CUDA 教训）
        return fn(q, k, v, None, 0.0, True, None)[0][0, 0, 0, 0].item()

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
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
