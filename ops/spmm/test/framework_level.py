# 应用层验证: GNN 特征传播 consumer（SpMM 消费位）双跑。
#
# Gowalla 训练链路的 SpMM 消费形态 = 图卷积特征传播 H' = Â @ H（issue
# #57: Torch-FL + FlagScale Gowalla 训练使用 CSR 稀疏×稠密）——本
# consumer 用 2 层传播复现该消费位（真机内存内缩样 4k 节点）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))

_CALLS = {"n": 0}


def _build_graph(dev, m=4096, k=32, density=0.01, seed=20261125):
    import torch
    g = torch.Generator().manual_seed(seed)  # CPU 生成后搬运（跨设备可复现）
    dense = (torch.rand(m, m, generator=g) < density).float().to(dev)
    # 行归一化邻接（Gowalla GCN 预处理形态）
    deg = dense.sum(1, keepdim=True).clamp_min(1.0)
    adj = dense / deg
    return adj.to_sparse_csr()


def _build_model(dev, dim=32):
    import torch

    class TwoLayerGCN:
        """2 层图卷积: H1 = σ(Â H0 W1), H2 = Â H1 W2（SpMM 消费位 ×2）。"""

        def __init__(self):
            torch.manual_seed(20261125)
            self.w1 = (torch.randn(dim, dim, device=dev) * 0.1
                       ).requires_grad_(True)
            self.w2 = (torch.randn(dim, dim, device=dev) * 0.1
                       ).requires_grad_(True)

        def __call__(self, adj, x):
            h = torch.nn.functional.relu(torch.sparse.mm(adj, x) @ self.w1)
            return torch.sparse.mm(adj, h) @ self.w2

    return TwoLayerGCN()


def _propagate(model, adj, x, steps):
    import torch

    with torch.no_grad():
        cur = x
        for _ in range(steps):
            cur = model(adj, cur)
    return cur


def run(profile):
    import torch

    from register import register_a1

    dev = profile.torch_device
    adj = _build_graph(dev)
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = torch.randn(4096, 32, generator=generator).to(dev)

    model = _build_model(dev)

    # baseline
    with torch.no_grad():
        base_out = model(adj, x)
    base_cont = _propagate(model, adj, x, 2)

    # injected
    _CALLS["n"] = 0
    lib = register_a1(profile.dispatch_key, counter=_CALLS)
    with torch.no_grad():
        inj_out = model(adj, x)
    inj_cont = _propagate(model, adj, x, 2)

    diff = (inj_out.float() - base_out.float()).abs().max().item()
    # 实现替换型算子（cusparse → index_add 组合）: 累加序不同 →
    # 容差断言不追位级（docs/testing.md 断言策略: 自定义数值实现档）
    assert diff < 1e-5, f"GCN 输出 diff={diff}"
    cont_diff = (inj_cont.float() - base_cont.float()).abs().max().item()
    assert cont_diff < 1e-4, f"迭代传播 diff={cont_diff}"
    # 2 层 × (1 前向 + 2 传播步) = ≥6 次 SpMM
    assert _CALLS["n"] >= 6, f"A1 计数 {_CALLS['n']} < 6（SpMM 未被真实消费）"

    # 训练语义: 反向不破坏（loss 对 W1 的梯度有限）
    out = model(adj, x)
    loss = out.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(model.w1).all()

    return {"ok": True,
            "consumer": "2-layer GCN feature propagation (fp32, 4k nodes)",
            "out_diff": diff,
            "propagation_match": True,
            "spmm_calls": _CALLS["n"]}


if __name__ == "__main__":
    from _profile import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1 else "thead-ppu")))
