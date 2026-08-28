# kernel 层直测: 精度 / 哨兵 / 性能（返回 metrics 供 run.py 落盘）。
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))                 # 引用同算子的 kernel/reference
sys.path.insert(0, str(OP_DIR.parent.parent))   # 仓库根（common 等）


def run(profile):
    import time

    import torch

    from kernel.triton_level import my_op_triton
    from reference import my_op_reference

    dev = profile.torch_device
    max_err = 0.0

    # 1) 精度: 多 shape × dtype vs 参考（容差随 dtype，逐组断言）
    for shape in [(64, 1024), (1, 14336), (8, 5000)]:   # 含非整倍数 N（#15）
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(*shape, dtype=dt, device=dev) * 2
            g = torch.randn(*shape, dtype=dt, device=dev)
            tol = 1e-5 if dt == torch.float32 else 1e-2
            err = (my_op_triton(x, g).float()
                   - my_op_reference(x, g).float()).abs().max().item()
            assert err < tol, f"{shape} {dt} err={err}"
            max_err = max(max_err, err)

    # 2) 哨兵: 确定性 + 输入敏感（抓静默 no-op 类缺陷，#11/#15）
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=dev)
    g = torch.randn_like(x)
    o1 = my_op_triton(x, g)
    assert torch.equal(o1, my_op_triton(x, g)), "同输入两次调用不一致"
    assert not torch.equal(o1, my_op_triton(x + 1, g)), "输出不随输入变化"

    # 3) 性能: 短采样（≤100 次；回归走 script/bench_perf 或 perf_registry）
    x = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev) * 2
    g = torch.randn_like(x)
    for _ in range(20):
        my_op_triton(x, g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        my_op_triton(x, g)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / 100 * 1000

    return {"ok": True, "max_err": round(max_err, 8),
            "sentinel": "deterministic+sensitive",
            "latency_ms": round(t_ms, 4)}


def perf_cases(profile):
    from common.perf import PerfCase

    def make(p):
        import torch
        x = torch.randn(8192, 8192, dtype=torch.bfloat16,
                        device=p.torch_device) * 2
        g = torch.randn(8192, 8192, dtype=torch.bfloat16,
                        device=p.torch_device)
        from kernel.triton_level import my_op_triton
        return lambda: my_op_triton(x, g)

    return [PerfCase("example.my_op.triton", group="example",
                     level="kernel", make_fn=make)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
