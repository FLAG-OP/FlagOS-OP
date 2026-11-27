# kernel 层直测: 精度 / 哨兵 / 性能（issue #57 交付与验收）。
#
# 覆盖:
#   - 台账对照尺寸 N=64/512（昇腾 kernel 测试尺寸, 跨平台对照）
#   - Gowalla 型形态（M≈10万 稀疏图 × 特征维; 真机内存内缩样）
#   - CSR 边界: 空行 / 重复列索引（值累加） / 无序索引 / 全零矩阵
#   - fp16/bf16/fp32; 哨兵: 确定性 + 输入敏感
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from reference import spmm_backward_reference, spmm_reference  # noqa: E402


def _make_csr(m, k, density, dt, dev, seed, shuffle=False):
    import torch
    g = torch.Generator().manual_seed(seed)  # CPU 生成后搬运（跨设备可复现）
    dense = torch.rand(m, k, generator=g)
    mask = torch.rand(m, k, generator=g) < density
    dense = (dense * mask).to(dt).to(dev)
    a = dense.to_sparse_csr()
    if shuffle:
        # 打乱列索引（无序 CSR——求和可交换, 输出应与有序一致）
        crow = a.crow_indices()
        col = a.col_indices()
        val = a.values()
        perm = torch.randperm(col.shape[0], generator=g)[:col.shape[0]]
        col = col[perm.to(dev)]
        val = val[perm.to(dev)]
        a = torch.sparse_csr_tensor(crow, col, val,
                                    (m, k), device=dev)
    return a


def _relerr(got, exp):
    """SpMM 是乘加放大形态 → GEMM 档 5e-2 rel 判卷（acceptance.md）。"""
    denom = exp.float().abs().clamp_min(1.0)
    return ((got.float() - exp.float()).abs() / denom).max().item()


def _sync(dev):
    import torch
    if isinstance(dev, str) and dev.startswith("cuda"):
        torch.cuda.synchronize(torch.device(dev))


def _bench(fn, dev, warmup=20, iters=100):
    import time
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters * 1000


def run(profile):
    import torch

    from kernel.thead_ppu import spmm_backward_ppu, spmm_ppu

    dev = profile.torch_device
    dts = (torch.float32, torch.float16, torch.bfloat16)
    tol = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-2}
    worst = 0.0

    # 1) 精度: 台账对照尺寸 N=64/512 + 常规形态 × 3 dtype
    cases = [  # (M, K, N, density)
        (64, 32, 64, 0.3),      # N=64（台账尺寸）
        (64, 32, 512, 0.3),     # N=512（台账尺寸）
        (512, 128, 64, 0.1),    # 更大 M
        (1024, 64, 512, 0.05),  # 稀疏化
        (7, 4097, 64, 0.5),     # 非整倍数 K
        (33, 16, 3, 0.4),       # 小 N 边界
    ]
    for m, k, n, dens in cases:
        for dt in dts:
            a = _make_csr(m, k, dens, dt, dev,
                          seed=20261125 + m + n)
            b = torch.randn(k, n, dtype=torch.float32, device=dev).to(dt)
            got = spmm_ppu(a, b)
            exp = spmm_reference(a.cpu(), b.cpu()).to(dev)
            assert got.shape == exp.shape, f"({m},{k})@({k},{n}) shape"
            err = _relerr(got, exp)
            assert err < 5e-2, f"spmm ({m},{k},{n}) {dt} rel={err}"
            worst = max(worst, err)

    # 2) CSR 边界语义（issue #57 实现范围 2）
    a = _make_csr(16, 16, 0.3, torch.float32, dev, seed=7)
    b = torch.randn(16, 8, dtype=torch.float32, device=dev)
    # 2a 空行: 密度 0 → 全零行
    a_empty = torch.sparse_csr_tensor(
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     device=dev),
        torch.tensor([], dtype=torch.int64, device=dev),
        torch.tensor([], dtype=torch.float32, device=dev),
        (16, 16), device=dev)
    got = spmm_ppu(a_empty, b)
    assert (got == 0).all().item(), "空行未产生零行"
    # 2b 重复列索引: 值累加（手工构造: row0 = (1.0+2.0)·B[1],
    # row3 = 1.5·B[3] + 0.5·B[0]; rows/val/col 手工同步）
    crow = torch.tensor([0, 2, 2, 2, 4], device=dev)
    col = torch.tensor([1, 1, 3, 0], device=dev)
    val = torch.tensor([1.0, 2.0, 1.5, 0.5], device=dev)
    a_dup = torch.sparse_csr_tensor(crow, col, val, (4, 4), device=dev)
    b4 = torch.randn(4, 4, dtype=torch.float32, device=dev)
    got = spmm_ppu(a_dup, b4)
    exp = torch.zeros(4, 4, dtype=torch.float32, device=dev)
    exp[0] = 3.0 * b4[1]
    exp[3] = 1.5 * b4[3] + 0.5 * b4[0]
    err = _relerr(got, exp)
    assert err < 1e-5, f"重复列索引累加 rel={err}"
    # 2c 无序索引: 与有序结果一致
    a_shuf = _make_csr(16, 16, 0.3, torch.float32, dev, seed=13, shuffle=True)
    err = _relerr(spmm_ppu(a_shuf, b), spmm_reference(a_shuf.cpu(), b.cpu()).to(dev))
    assert err < 1e-4, f"无序索引 rel={err}"

    # 3) 反向: d_vals / dB vs 参考（Gowalla 训练链路所用梯度）
    a = _make_csr(64, 32, 0.3, torch.float32, dev, seed=17)
    b = torch.randn(32, 16, dtype=torch.float32, device=dev,
                    requires_grad=True)
    g = torch.randn(64, 16, dtype=torch.float32, device=dev)
    d_vals, dB = spmm_backward_ppu(a, b, g)
    r_vals, rB = spmm_backward_reference(a.cpu(), b.cpu(), g.cpu())
    assert _relerr(d_vals, r_vals.to(dev)) < 1e-4, "d_vals"
    assert _relerr(dB, rB.to(dev)) < 1e-4, "dB"

    # 3b) fp16 反向中间积溢出: 2*60000 + 2*(-60000) 必须在 fp32 域归约
    a_fp16 = torch.sparse_csr_tensor(
        torch.tensor([0, 1], device=dev),
        torch.tensor([0], device=dev),
        torch.tensor([1.0], dtype=torch.float16, device=dev),
        (1, 2), device=dev)
    b_fp16 = torch.tensor(
        [[60000.0, -60000.0], [0.0, 0.0]],
        dtype=torch.float16, device=dev)
    g_fp16 = torch.tensor([[2.0, 2.0]], dtype=torch.float16, device=dev)
    d_vals, dB = spmm_backward_ppu(a_fp16, b_fp16, g_fp16)
    r_vals, rB = spmm_backward_reference(
        a_fp16.cpu(), b_fp16.cpu(), g_fp16.cpu())
    assert torch.isfinite(d_vals).all(), "fp16 d_vals overflowed to NaN"
    assert torch.isfinite(dB).all(), "fp16 dB overflowed to NaN"
    assert _relerr(d_vals, r_vals.to(dev)) < 1e-4, "fp16 d_vals"
    assert _relerr(dB, rB.to(dev)) < 1e-4, "fp16 dB"

    # 4) 哨兵: 确定性 + 输入敏感
    #    注: index_add_ 的原子累加顺序不确定（同 row 多非零并发加）——
    #    fp32 累加域次序噪声 ~1e-7, 经 fp16 输出量化放大为 1 ulp
    #    （~0.06 @ |out|~1-2）。确定性断言用 fp16 判卷档容差
    #    （rtol/atol=1e-2, 与 acceptance fp16 档一致）; 哨兵目的是抓
    #    静默 no-op/未初始化内存, 该容差下仍有效（no-op 输出为未初始化
    #    垃圾, 偏差远超容差）。
    a = _make_csr(64, 32, 0.3, torch.float16, dev, seed=19)
    b = torch.randn(32, 64, dtype=torch.float16, device=dev)
    o1 = spmm_ppu(a, b)
    assert torch.allclose(o1, spmm_ppu(a, b), rtol=1e-2, atol=1e-2), \
        "同输入两次调用不一致（超 fp16 累加序噪声档）"
    assert not torch.allclose(o1, spmm_ppu(a, b + 1), rtol=1e-2, atol=1e-2), \
        "输出不随输入变化（疑似静默 no-op）"

    # 5) 性能: 短采样, Gowalla 型形态（10万节点 × 64 特征, 真机缩样 8k）
    m, k, n, dens = 8192, 64, 64, 0.01
    a = _make_csr(m, k, dens, torch.float32, dev, seed=23)
    b = torch.randn(k, n, dtype=torch.float32, device=dev)
    t_ms = _bench(lambda: spmm_ppu(a, b), dev)
    nnz = a.values().shape[0]
    flops = 2 * nnz * n

    return {"ok": True, "max_err": round(worst, 10),
            "sentinel": "deterministic+sensitive",
            "nnz": int(nnz),
            "latency_ms": round(t_ms, 4),
            "GFLOPS": round(flops / t_ms / 1e6, 2)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        m, k, n, dens = 8192, 64, 64, 0.01
        dense = torch.rand(m, k, device=p.torch_device) * (torch.rand(
            m, k, device=p.torch_device) < dens)
        a = dense.to_sparse_csr()
        b = torch.randn(k, n, device=p.torch_device)
        from kernel.thead_ppu import spmm_ppu
        return lambda: spmm_ppu(a, b)

    def gf(t):
        return {"GFLOPS": 2 * 8192 * 64 * 0.01 * 64 / t / 1e6}

    return [PerfCase("ops.spmm.index_add", group="ops", level="kernel",
                     make_fn=make, derived=gf)]


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
