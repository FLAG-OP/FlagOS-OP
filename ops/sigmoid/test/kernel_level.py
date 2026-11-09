"""kernel 层：尺寸、dtype、布局、边界、哨兵及性能烟测。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


def _sync(device: str) -> None:
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mlu"):
        import torch_mlu  # noqa: F401
        torch.mlu.synchronize()


def _bench(fn, device: str, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _record_error(got, expected, current_abs: float,
                  current_rel: float) -> tuple[float, float]:
    import torch
    finite = torch.isfinite(expected.float())
    if not bool(finite.any()):
        return current_abs, current_rel
    diff = (got.float()[finite] - expected.float()[finite]).abs()
    denom = expected.float()[finite].abs().clamp_min(1e-6)
    return max(current_abs, diff.max().item()), \
        max(current_rel, (diff / denom).max().item())


def run(profile):
    import torch

    try:
        from ops.sigmoid.kernel.torch_level import sigmoid_torch
        from ops.sigmoid.kernel.triton_level import sigmoid_triton
        from ops.sigmoid.reference import sigmoid_reference
    except ImportError:
        from kernel.torch_level import sigmoid_torch
        from kernel.triton_level import sigmoid_triton
        from reference import sigmoid_reference

    device = profile.torch_device
    tolerances = {
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (8e-3, 8e-3),
        torch.float32: (2e-6, 2e-5),
    }

    # 覆盖极小输入、两类调度边界、非整倍数长尾和真实模型 rank。
    shapes = [
        (), (1,), (17,),
        (255,), (256,), (257,),
        (511,), (512,), (513,),
        (65535,), (65536,), (65537,),
        (64, 1024), (1, 14336), (8, 5000), (7, 4097),
        (2, 128, 768), (2, 64, 4096), (1, 4, 32, 512),
    ]
    max_abs = 0.0
    max_rel = 0.0
    cases = 0

    for shape in shapes:
        for dtype, (atol, rtol) in tolerances.items():
            x = torch.randn(shape, dtype=dtype, device=device) * 8
            got = sigmoid_triton(x)
            expected = sigmoid_reference(x)
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
            assert got.shape == x.shape and got.dtype == x.dtype
            max_abs, max_rel = _record_error(
                got, expected, max_abs, max_rel)
            cases += 1

    # 非连续布局：转置、步长切片、expand，三种核心 dtype 全覆盖。
    for dtype, (atol, rtol) in tolerances.items():
        layout_cases = [
            torch.randn(129, 67, dtype=dtype, device=device).t(),
            torch.randn(128, 130, dtype=dtype, device=device)[::2, 1::3],
            torch.randn(1, 257, dtype=dtype, device=device).expand(8, 257),
        ]
        for x in layout_cases:
            assert not x.is_contiguous()
            got, expected = sigmoid_triton(x), sigmoid_reference(x)
            torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
            max_abs, max_rel = _record_error(
                got, expected, max_abs, max_rel)
            cases += 1

    # 空张量与 IEEE 特殊值。
    ieee_values = [-float("inf"), -100.0, -20.0, -1.0, -0.0,
                   0.0, 1.0, 20.0, 100.0, float("inf"), float("nan")]
    for dtype, (atol, rtol) in tolerances.items():
        empty = torch.empty((0, 7), device=device, dtype=dtype)
        out = sigmoid_triton(empty)
        assert out.shape == empty.shape and out.dtype == dtype
        cases += 1

        special = torch.tensor(ieee_values, device=device, dtype=dtype)
        got, expected = sigmoid_triton(special), sigmoid_reference(special)
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol,
                                   equal_nan=True)
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        max_abs, max_rel = _record_error(got, expected, max_abs, max_rel)
        cases += 1

    # 非优化 dtype 的语义回退范围。
    for dtype in (torch.int32, torch.int64, torch.bool):
        if dtype == torch.bool:
            x = torch.tensor([False, True], dtype=dtype, device=device)
        else:
            x = torch.arange(-32, 32, dtype=dtype, device=device)
        got, expected = sigmoid_triton(x), sigmoid_reference(x)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, expected)
        cases += 1

    # 确定性、输入敏感性与 torch 级交叉验证。
    x = torch.randn(4097, dtype=torch.float32, device=device)
    first = sigmoid_triton(x)
    assert torch.equal(first, sigmoid_triton(x)), "同输入两次调用不一致"
    assert not torch.equal(first, sigmoid_triton(x + 1)), "输出不随输入变化"
    torch.testing.assert_close(first, sigmoid_torch(x), atol=2e-6, rtol=2e-5)

    # 单点性能只作烟测；完整尺寸×dtype 矩阵见 script/bench_matrix.py。
    perf_x = torch.randn(2048, 2048, dtype=torch.float32, device=device)
    triton_ms = _bench(lambda: sigmoid_triton(perf_x), device)
    native_ms = _bench(lambda: torch.sigmoid(perf_x), device)
    return {
        "ok": True,
        "cases": cases,
        "shape_count": len(shapes),
        "optimized_dtypes": ["float16", "bfloat16", "float32"],
        "fallback_dtypes": ["int32", "int64", "bool"],
        "max_abs": max_abs,
        "max_rel": max_rel,
        "sentinel": "deterministic+sensitive",
        "triton_ms": round(triton_ms, 6),
        "native_ms": round(native_ms, 6),
        "speedup_vs_native": round(native_ms / triton_ms, 4),
    }


def perf_cases(profile):
    from common.perf import PerfCase

    def make(fn):
        def _make(p):
            import torch
            x = torch.randn(2048, 2048, dtype=torch.float32,
                            device=p.torch_device)
            return lambda: fn(x)
        return _make

    def bandwidth(latency_ms):
        return {"GBps": 2048 * 2048 * 4 * 2 / latency_ms / 1e6}

    try:
        from ops.sigmoid.kernel.triton_level import sigmoid_triton
    except ImportError:
        from kernel.triton_level import sigmoid_triton
    return [PerfCase("ops.sigmoid.triton", group="ops", level="kernel",
                     make_fn=make(sigmoid_triton), derived=bandwidth)]


if __name__ == "__main__":
    from common.device import load_profile
    print(run(load_profile(sys.argv[1] if len(sys.argv) > 1
                           else "p800-kunlunxin")))
