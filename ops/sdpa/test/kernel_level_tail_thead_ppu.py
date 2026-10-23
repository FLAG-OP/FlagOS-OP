# thead-ppu 专用 kernel 尾块/特殊分布补充直测（任务书 §2d 覆盖矩阵）。
#
# test/kernel_level.py（平台无关 36 case）已在 thead-ppu 全绿；本文件补:
#   - ≥3 shape 含 3072 / 5000 / 5120 尾块维（含 5000×5001 + bool mask,
#     专门打到 PPU efficient kernel 的 attn_bias 16B stride 对齐修复路径）
#   - fp32 / fp16 / bf16 × ≥3 seed
#   - zeros / large / boundary 特殊分布（docs/acceptance.md 特殊用例口径）
# 平台无关共享资产（kernel_level.py / reference.py / goldendata）零改动。
#
# 运行: python3 test/kernel_level_tail_thead_ppu.py [thead-ppu]
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

TOL = {"float32": 1e-5, "float16": 2e-2, "bfloat16": 2e-2}


def _gen(shape, dt, seed, kind, dev):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "zeros":
        t = torch.zeros(*shape)
    elif kind == "large":
        # ±1e4: naive softmax 的 exp(score) 会溢出（饱和边界, 见 acceptance）
        t = (torch.rand(*shape, generator=g) * 2 - 1) * 1e4
    elif kind == "boundary":
        # dtype 上限/2（boundary 口径）
        t = torch.randn(*shape, generator=g)
    else:
        t = torch.randn(*shape, generator=g) * 0.5
    t = t.to(dt)
    if kind == "boundary":
        t = t / t.abs().max().clamp_min(1e-30) * (torch.finfo(dt).max / 2)
    return t.to(dev)


def _case(tag, B, Hq, Hkv, Sq, Skv, D, dt, seed, causal, mask_kind, kind,
          dev, results):
    import torch

    from kernel.triton_level import sdpa_triton
    from reference import sdpa_reference

    q = _gen((B, Hq, Sq, D), dt, seed, kind, dev)
    k = _gen((B, Hkv, Skv, D), dt, seed, kind, dev)
    v = _gen((B, Hkv, Skv, D), dt, seed, kind, dev)
    if kind == "boundary":
        # 输出侧边界: V 顶到 dtype 上限/2, q/k 保持小值（避免 scores 无意义
        # 的全饱和; exp 溢出边界由 large 分布覆盖）
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

    out = sdpa_triton(q, k, v, m, 0.0, causal, None, Hq != Hkv)
    ref = sdpa_reference(
        q.cpu(), k.cpu(), v.cpu(),
        m.cpu() if m is not None else None, 0.0, causal, None, Hq != Hkv
    ).to(dev).float()
    diff = (out.float() - ref).abs()
    nan_rows = torch.isnan(ref[..., 0])
    if nan_rows.any():
        keep = ~nan_rows.unsqueeze(-1).expand_as(diff)
        diff = diff[keep]
    finite = torch.isfinite(diff)
    err = diff[finite].max().item() if finite.any() else float("inf")
    n_nonfinite = int((~torch.isfinite(out)).sum())

    name = f"{str(dt).split('.')[-1]:9s} seed{seed}"
    line = (f"  [{tag}] {name} S{Sq}/{Skv} D{D} causal={int(causal)} "
            f"mask={mask_kind or '-':5s} kind={kind:8s} err={err:.3e}")
    dtn = str(dt).split(".")[-1]

    if kind == "boundary":
        # boundary 判定: 输出量级≈dtype 上限/2, 绝对容差无意义——按黄金
        # extreme 口径改相对容差 5e-3（goldendata/inputs_spec.yaml
        # tolerance.relative）。
        both = torch.isfinite(out) & torch.isfinite(ref)
        rel = ((out[both] - ref[both]).abs()
               / ref[both].abs().clamp_min(1e-3)).max().item() if \
            bool(both.any()) else float("inf")
        if dtn == "bfloat16":
            # 平台分化（PLATFORM.md thead 节）: 厂商 kernel 的 bf16 归约在
            # V≈dtype_max/2 × 长序列下中间量溢出; 原生 F.sdpa 同样溢出。
            # 判定 = 与原生行为一致 + 有限子集相对精度。
            import torch.nn.functional as F
            nat = F.scaled_dot_product_attention(q, k, v, None,
                                                is_causal=causal)
            parity = bool((torch.isinf(out) == torch.isinf(nat)).all())
            ok = parity and rel < 5e-3
            line += (f" rel={rel:.3e} nonfinite={n_nonfinite} "
                     f"parity_vs_native={parity}")
        else:
            ok = rel < 5e-3 and n_nonfinite == 0
            line += f" rel={rel:.3e} nonfinite={n_nonfinite}"
        results.append(line + ("  OK" if ok else "  <== FAIL"))
        assert ok, line
        return

    tol = TOL[dtn]
    status = "OK" if (err < tol and n_nonfinite == 0) else "FAIL"
    line += f" nonfinite={n_nonfinite}  {status}" + (
        "  <== 超容差" if status == "FAIL" else "")
    results.append(line)
    assert status == "OK", line


def run(profile):
    import torch

    dev = profile.torch_device
    if profile.name != "thead-ppu":
        raise ValueError("本文件是 thead-ppu 专用补充直测")
    results: list[str] = []
    t0 = time.time()

    # 1) 尾块 shape 矩阵: 3072 / 5000 / 5120 × 3 dtype × 3 seed
    tail_shapes = [
        # (B,Hq,Hkv,Sq,Skv,D,causal,mask,tag)
        (1, 4, 4, 3072, 3072, 64, True, None, "s3072-causal"),
        (1, 4, 4, 5000, 5120, 64, False, None, "s5000x5120"),
        (1, 4, 4, 5000, 5001, 64, False, "bool", "s5000x5001-mask"),
    ]
    for (B, Hq, Hkv, Sq, Skv, D, causal, mask, tag) in tail_shapes:
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            for seed in (1, 2, 3):
                _case(tag, B, Hq, Hkv, Sq, Skv, D, dt, seed, causal, mask,
                      "normal", dev, results)

    # 2) 特殊分布: zeros / large / boundary（S=1024, 与 exp 溢出/截断口径对齐）
    for kind in ("zeros", "large", "boundary"):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            _case(f"special-{kind}", 1, 4, 4, 1024, 1024, 64, dt, 5,
                  True, None, kind, dev, results)

    # 3) 哨兵: 尾块 shape 下确定性 + 输入敏感（#11 静默 no-op 类）
    from kernel.triton_level import sdpa_triton

    q, k, v = (_gen((1, 4, 3072, 64), torch.float16, 7, "normal", dev)
               for _ in range(3))
    o1 = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    o2 = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    assert torch.equal(o1, o2), "同输入两次调用不一致（静默 no-op?）"
    o3 = sdpa_triton(q + 0.25, k, v, None, 0.0, True, None, False)
    assert not torch.equal(o1, o3), "输出不随输入变化"
    results.append("  [OK] 哨兵(3072 尾块): 确定性 + 输入敏感")

    for ln in results:
        print(ln)
    summary = {"ok": True, "cases": len(results),
               "seconds": round(time.time() - t0, 1)}
    print(summary)
    return summary


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
