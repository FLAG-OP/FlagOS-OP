# thead-ppu 专用 kernel 尾块/特殊分布补充直测（任务书 §2d 覆盖矩阵）。
#
# test/kernel_level.py（平台无关 30 case）已在 thead-ppu 全绿；本文件补:
#   - ≥3 shape 含 3072 / 5000 / 5120 尾块维（embedding 无归约, 取 gather
#     长度维——#15a 尾块 masked-load 陷阱不适用, 但大长度仍打满 native
#     row-gather 的调度路径）
#   - fp32 / fp16 / bf16 × ≥3 seed
#   - zeros / large / boundary 特殊分布（docs/acceptance.md 特殊用例口径;
#     boundary 对 embedding 取词表边界 index 0/num_weights-1 + 权重量级
#     顶到 dtype 上限/2）
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
NUM_WEIGHTS = 4096
DIM = 128


def _gen_weight(shape, dt, seed, kind, dev):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "zeros":
        t = torch.zeros(*shape)
    elif kind == "large":
        # ±1e4: 权重量级饱和边界（acceptance 口径）
        t = (torch.rand(*shape, generator=g) * 2 - 1) * 1e4
    elif kind == "boundary":
        # dtype 上限/2（boundary 口径）
        t = torch.randn(*shape, generator=g)
        t = t / t.abs().max().clamp_min(1e-30) * (torch.finfo(dt).max / 2)
    else:
        t = torch.randn(*shape, generator=g) * 0.5
    return t.to(dt).to(dev)


def _gen_indices(shape, seed, kind, dev):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed + 17)
    if kind == "boundary":
        # 词表边界: 0 与 num_weights-1 交替（静默越界类缺陷哨兵, #4）
        n = 1
        for s in shape:
            n *= s
        vals = torch.arange(n) % NUM_WEIGHTS
        return vals.reshape(shape).to(dev)
    return torch.randint(0, NUM_WEIGHTS, shape, generator=g).to(dev)


def _case(tag, n_idx, dt, seed, kind, dev, results):
    import torch

    from kernel.platform import embedding
    from reference import embedding_reference

    weight = _gen_weight((NUM_WEIGHTS, DIM), dt, seed, kind, dev)
    indices = _gen_indices((n_idx,), seed, kind, dev)
    out = embedding(weight, indices)
    ref = embedding_reference(weight.cpu(), indices.cpu())
    diff = (out.float().cpu() - ref.float()).abs()
    err = diff.max().item() if diff.numel() else 0.0
    n_nonfinite = int((~torch.isfinite(out)).sum())
    dtn = str(dt).split(".")[-1]

    line = (f"  [{tag}] {dtn:9s} seed{seed} n_idx={n_idx} kind={kind:8s} "
            f"err={err:.3e}")

    if kind == "boundary":
        # boundary 判定: 输出量级≈dtype 上限/2, 绝对容差无意义——按黄金
        # extreme 口径改相对容差 5e-3（goldendata/inputs_spec.yaml 同源）。
        both = torch.isfinite(out.float().cpu()) & torch.isfinite(ref.float())
        rel = (
            ((out.float().cpu()[both] - ref.float()[both]).abs()
             / ref.float()[both].abs().clamp_min(1e-3)).max().item()
            if bool(both.any()) else float("inf")
        )
        ok = rel < 5e-3 and n_nonfinite == 0
        line += f" rel={rel:.3e} nonfinite={n_nonfinite}"
    else:
        tol = TOL[dtn]
        ok = err < tol and n_nonfinite == 0
        line += f" nonfinite={n_nonfinite}"

    results.append(line + ("  OK" if ok else "  <== FAIL"))
    assert ok, line


def run(profile):
    import torch

    dev = profile.torch_device
    if profile.name != "thead-ppu":
        raise ValueError("本文件是 thead-ppu 专用补充直测")
    results: list[str] = []
    t0 = time.time()

    # 1) 尾块 shape 矩阵: 3072 / 5000 / 5120 gather 长度 × 3 dtype × 3 seed
    for n_idx, tag in ((3072, "n3072"), (5000, "n5000"), (5120, "n5120")):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            for seed in (1, 2, 3):
                _case(tag, n_idx, dt, seed, "normal", dev, results)

    # 2) 特殊分布: zeros / large / boundary（gather 长度 4096, 与饱和/
    #    词表边界口径对齐）
    for kind in ("zeros", "large", "boundary"):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            _case(f"special-{kind}", 4096, dt, 5, kind, dev, results)

    # 3) 哨兵: 大 shape 下确定性 + 输入敏感（#11 静默 no-op 类）
    from kernel.platform import embedding

    weight = _gen_weight((NUM_WEIGHTS, DIM), torch.float16, 7, "normal", dev)
    indices = _gen_indices((3072,), 7, "normal", dev)
    o1 = embedding(weight, indices)
    o2 = embedding(weight, indices)
    assert torch.equal(o1, o2), "同输入两次调用不一致（静默 no-op?）"
    o3 = embedding(weight + 0.25, indices)
    assert not torch.equal(o1, o3), "输出不随输入变化"
    results.append("  [OK] 哨兵(3072 长度): 确定性 + 输入敏感")

    for ln in results:
        print(ln)
    summary = {"ok": True, "cases": len(results),
               "seconds": round(time.time() - t0, 1)}
    print(summary)
    return summary


if __name__ == "__main__":
    from _profile import load_profile

    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else None)))
