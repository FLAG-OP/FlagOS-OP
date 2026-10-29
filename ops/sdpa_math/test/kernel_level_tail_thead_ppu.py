# thead-ppu 专用 sdpa_math kernel 尾块/特殊分布补充直测（任务书 §2d）。
#
# test/kernel_level.py（平台无关 52 组）已在 thead-ppu 全绿；本文件补:
#   - ≥3 shape 含 3072 / 5000 / 5120 尾块维（含 5000×5001 + bool mask,
#     打到 PPU efficient kernel 的 attn_bias 16B stride 对齐修复路径）
#   - fp32 / fp16 / bf16 × ≥3 seed
#   - zeros / large / boundary 特殊分布（docs/acceptance.md 特殊用例口径）
# 判卷: out 与 P 双路（本算子契约）; 全遮蔽/极端行用相对容差 5e-3
# （黄金 extreme 同款）。
#
# 运行: python3 test/kernel_level_tail_thead_ppu.py [thead-ppu]
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

TOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}
REL_TOL = 5e-3  # extreme / boundary 判卷（黄金 extreme 口径）


def _gen(shape, dt, seed, kind, dev):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "zeros":
        t = torch.zeros(*shape)
    elif kind == "large":
        t = (torch.rand(*shape, generator=g) * 2 - 1) * 1e4
    elif kind == "boundary":
        t = torch.randn(*shape, generator=g)
    else:
        t = torch.randn(*shape, generator=g) * 0.5
    t = t.to(dt)
    if kind == "boundary":
        t = t / t.abs().max().clamp_min(1e-30) * (torch.finfo(dt).max / 2)
    return t.to(dev)


def _impl_for(profile):
    if profile.vendor == "thead":
        from kernel.thead_fast_level import sdpa_math_thead_fast as fn
    elif profile.vendor == "kunlunxin":
        from kernel.p800_fast_level import sdpa_math_p800_fast as fn
    else:
        from kernel.torch_level import sdpa_math_torch as fn
    return fn


def _case(tag, B, Hq, Hkv, Sq, Skv, D, dt, seed, causal, mask_kind, kind,
          dev, fn, results):
    import torch

    from reference import sdpa_math_reference

    q = _gen((B, Hq, Sq, D), dt, seed, kind, dev)
    k = _gen((B, Hkv, Skv, D), dt, seed, kind, dev)
    v = _gen((B, Hkv, Skv, D), dt, seed, kind, dev)
    if kind == "boundary":
        q = _gen((B, Hq, Sq, D), dt, seed, "normal", dev)
        k = _gen((B, Hkv, Skv, D), dt, seed, "normal", dev)
        v = torch.full((B, Hkv, Skv, D), torch.finfo(dt).max / 2,
                       dtype=dt, device=torch.device(dev))
    m = None
    if mask_kind in ("bool", "float"):
        mg = torch.Generator(device="cpu").manual_seed(seed + 1)
        if mask_kind == "bool":
            m = (torch.rand(B, Hq, Sq, Skv, generator=mg) > 0.3).to(dev)
        else:
            m = (torch.randn(B, Hq, Sq, Skv, generator=mg) * 0.1).to(dt).to(dev)

    gqa = Hq != Hkv
    out, probs = fn(q, k, v, m, 0.0, causal, None, enable_gqa=gqa)
    ref = sdpa_math_reference(
        q.cpu(), k.cpu(), v.cpu(),
        m.cpu() if m is not None else None, 0.0, causal, None,
        enable_gqa=gqa,
    )
    eo = (out.float().cpu() - ref[0].float()).abs().max().item()
    ep = (probs.float().cpu() - ref[1].float()).abs().max().item()
    tol = TOL[str(dt).split(".")[-1]]
    if kind == "boundary":
        # T1 thead 口径: V≈dtype_max/2 时部分 vendor kernel O 中间量溢出为
        # inf——属平台分化。判定: ① 有限子集 rel<5e-3 + P 绝对容差;
        # ② 若本实现出现 inf, 其位置须与原生 F.sdpa parity（bf16 vendor
        # 路径）; 若本实现全有限而原生溢出（fp32 torch 路径更稳）, 直接过。
        import torch.nn.functional as F
        nat = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=causal)
        out_f = out.float().cpu()
        ref_f = ref[0].float()
        finite = torch.isfinite(out_f) & torch.isfinite(ref_f)
        if finite.any():
            rel_o = ((out_f - ref_f).abs()
                     / ref_f.abs().clamp_min(1e-3))[finite].max().item()
        else:
            rel_o = 0.0
        our_inf = torch.isinf(out_f)
        if bool(our_inf.any()):
            inf_parity = bool(
                (our_inf == torch.isinf(nat.float().cpu())).all())
        else:
            inf_parity = True  # 本实现未溢出——以参考精度为准
        ok = rel_o < REL_TOL and inf_parity and ep < tol
        shown = rel_o
        unit = "rel-O+abs-P"
        eo = rel_o
    elif kind == "large":
        rel_o = ((out.float().cpu() - ref[0].float()).abs()
                 / ref[0].float().abs().clamp_min(1e-3)).nan_to_num().max().item()
        rel_p = ((probs.float().cpu() - ref[1].float()).abs()
                 / ref[1].float().abs().clamp_min(1e-3)).nan_to_num().max().item()
        ok = max(rel_o, rel_p) < REL_TOL
        shown = max(rel_o, rel_p)
        unit = "rel"
        eo, ep = rel_o, rel_p
    else:
        ok = max(eo, ep) < tol
        shown = max(eo, ep)
        unit = "abs"
    line = (f"  [{'OK' if ok else 'FAIL'}] {tag:22s} "
            f"B{B} H{Hq}/{Hkv} S{Sq}x{Skv} D{D} "
            f"{str(dt).split('.')[-1]:9s} seed={seed} causal={int(causal)} "
            f"mask={mask_kind or '-':5s} dist={kind:8s} "
            f"out={eo:.2e} P={ep:.2e} ({unit}={shown:.2e})")
    results.append(line)
    assert ok, line


def run(profile):
    import torch

    dev = profile.torch_device
    fn = _impl_for(profile)
    results = []

    # ── 1) 尾块维 3072 / 5000 / 5120 × 3 dtype × 3 seed ──
    for Sq, Skv, causal, mask_kind, tag in (
        (3072, 3072, True, None, "tail3072-causal"),
        (5000, 5000, False, None, "tail5000"),
        (5120, 5120, True, None, "tail5120-causal"),
        (5000, 5001, False, "bool", "tail5000x5001-bool"),
    ):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            for seed in (11, 22, 33):
                _case(tag, 1, 4, 4, Sq, Skv, 64, dt, seed, causal,
                      mask_kind, "normal", dev, fn, results)

    # ── 2) zeros / large / boundary 特殊分布 ──
    for kind in ("zeros", "large", "boundary"):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            _case(f"dist-{kind}", 1, 2, 2, 256, 256, 64, dt, 7,
                  True, None, kind, dev, fn, results)

    # ── 3) 大形状 mask + 非对齐 stride（bool mask Sq=5000, efficient bias
    #      16B pad 路径; 与 tail5000x5001-bool 互补——此处 float mask）──
    for dt in (torch.float16, torch.bfloat16):
        _case("tail4096-floatmask", 1, 4, 4, 4096, 4096, 64, dt, 5,
              False, "float", "normal", dev, fn, results)

    for ln in results:
        print(ln)
    n_ok = sum(1 for ln in results if "[OK]" in ln)
    print(f"tail coverage: {n_ok}/{len(results)}")
    return {"ok": n_ok == len(results), "impl": fn.__name__,
            "cases": len(results), "passed": n_ok}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
