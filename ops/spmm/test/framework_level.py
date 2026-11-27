# 应用层验证: GNN 特征传播 consumer（SpMM 消费位）双跑。
#
# Gowalla 训练链路的 SpMM 消费形态 = 图卷积特征传播 H' = Â @ H（issue
# #57: Torch-FL + FlagScale Gowalla 训练使用 CSR 稀疏×稠密）——本
# consumer 用 2 层传播复现该消费位（真机内存内缩样 4k 节点）。
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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


def _run_mode(profile, mode):
    import torch

    from register import register_a1

    dev = profile.torch_device
    adj = _build_graph(dev)
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = torch.randn(4096, 32, generator=generator).to(dev)

    model = _build_model(dev)
    registered = False
    calls = 0
    if mode == "injected":
        lib = register_a1(profile.dispatch_key, counter=_CALLS)
        assert lib is not None
        registered = True
        calls = _CALLS["n"]

    with torch.no_grad():
        output = model(adj, x).detach().cpu()

    model = _build_model(dev)
    out = model(adj, x)
    loss = out.float().pow(2).mean()
    loss.backward()
    grads = [model.w1.grad, model.w2.grad]
    grad_finite = [bool(g is not None and torch.isfinite(g).all()) for g in grads]
    assert all(grad_finite), f"{mode} gradients are missing or non-finite"

    output_path = Path(tempfile.gettempdir()) / (
        f"spmm-{mode}-{id(profile)}-output.pt")
    grads_path = Path(tempfile.gettempdir()) / (
        f"spmm-{mode}-{id(profile)}-grads.pt")
    torch.save(output, output_path)
    torch.save(grads, grads_path)
    return {
        "mode": mode,
        "registered": registered,
        "calls": calls,
        "output_path": str(output_path),
        "grads_path": str(grads_path),
    }


def _run_worker(profile, mode):
    command = [sys.executable, str(Path(__file__).resolve()),
               profile.name, mode]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def run(profile):
    import torch

    try:
        baseline = _run_worker(profile, "baseline")
        injected = _run_worker(profile, "injected")
        assert baseline["registered"] is False and baseline["calls"] == 0, \
            "baseline process must not contain the A1 registration"
        assert injected["registered"] is True and injected["calls"] >= 6, \
            f"injected SpMM calls={injected['calls']} < 6"

        base_out = torch.load(baseline["output_path"])
        inj_out = torch.load(injected["output_path"])
        base_grads = torch.load(baseline["grads_path"])
        inj_grads = torch.load(injected["grads_path"])
        diff = (inj_out.float() - base_out.float()).abs().max().item()
        # 实现替换型算子（cusparse → index_add 组合）: 累加序不同 →
        # 容差断言不追位级（docs/testing.md 断言策略）
        assert diff < 1e-5, f"GCN output diff={diff}"
        grad_diffs = [(inj - base).abs().max().item()
                      for inj, base in zip(inj_grads, base_grads)]
        assert max(grad_diffs) < 1e-4, f"training gradient diff={grad_diffs}"
        assert all(torch.isfinite(grad).all()
                   for grad in base_grads + inj_grads), \
            "training gradients must be finite"

        return {"ok": True,
                "consumer": "2-layer GCN feature propagation (fp32, 4k nodes)",
                "out_diff": diff,
                "grad_diff": grad_diffs,
                "baseline_calls": baseline["calls"],
                "injected_calls": injected["calls"],
                "spmm_calls": injected["calls"]}
    finally:
        for worker in (baseline if "baseline" in locals() else None,
                       injected if "injected" in locals() else None):
            if worker is None:
                continue
            for key in ("output_path", "grads_path"):
                Path(worker[key]).unlink(missing_ok=True)


if __name__ == "__main__":
    from _profile import load_profile
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "thead-ppu"
    if len(sys.argv) > 2 and sys.argv[2] in ("baseline", "injected"):
        print(json.dumps(_run_mode(load_profile(profile_name), sys.argv[2])))
    else:
        print(run(load_profile(profile_name)))
