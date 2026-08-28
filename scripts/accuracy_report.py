#!/usr/bin/env python3
"""三方精度对比: 自研 kernel / FlagGems / PyTorch 原生（同输入同 fp32 参考）。

用法:
  python3 scripts/accuracy_report.py --device p800-kunlunxin

口径:
  - 参考: 同一输入 cast 到 fp32 后在设备上计算
  - pointwise: max abs err；GEMM: 相对 err（除以 max|ref|）
  - 容差线（本库约定）: fp32=1e-5 · bf16/fp16=1e-2 · GEMM 相对 5e-2
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POINTWISE_SHAPES = [(64, 1024), (1, 14336), (128, 5120), (8192, 2048)]
DTYPES = ["bfloat16", "float16", "float32"]
BMM_SHAPES = [(8, 256, 256, 256), (4, 128, 512, 64),
              (2, 1024, 512, 1024), (16, 64, 64, 64)]


def _load_example(rel: str):
    p = ROOT / "examples" / rel / "example.py"
    spec = importlib.util.spec_from_file_location(
        "acc_" + rel.replace("-", "_").replace("/", "_"), p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_grid(dev, impls, ref_fn, make_inputs, dts, relative=False):
    """返回 {dtype: max_err}，异常的 dtype 标 'ERR:...'。"""
    import torch
    out = {}
    for dt in dts:
        try:
            d = getattr(torch, dt)
            worst = 0.0
            for args in make_inputs(d):
                results = [f(*args) for f in impls]
                ref = ref_fn(*[a.float() for a in args])
                for r in results:
                    if relative:
                        scale = ref.abs().max().item() + 1e-6
                        e = (r.float() - ref).abs().max().item() / scale
                    else:
                        e = (r.float() - ref).abs().max().item()
                    worst = max(worst, e)
            out[dt] = worst
        except Exception as ex:
            out[dt] = f"ERR:{type(ex).__name__}"
    return out


def fmt(v, tol):
    if isinstance(v, str):
        return v
    mark = "✓" if v <= tol else "⚠"
    return f"{v:.2e} {mark}"


def main() -> int:
    from common.device import load_profile
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="p800-kunlunxin")
    args = ap.parse_args()
    profile = load_profile(args.device)
    dev = profile.torch_device
    print(f"[accuracy] {profile.summary()}\n")

    import torch
    import torch.nn.functional as F
    from flag_gems import ops
    from routes.a1_aten import register_aten as RA
    from routes.a2_dispatch.plugin import kernels as RK

    bmm_mod = _load_example("bmm-fullstack")
    smx_mod = _load_example("softmax-fullstack")
    K1 = RA._load_kernels()

    def pw_inputs(two=False):
        def make(d):
            for sh in POINTWISE_SHAPES:
                if two:
                    yield (torch.randn(*sh, dtype=d, device=dev) * 2,
                           torch.randn(*sh, dtype=d, device=dev))
                else:
                    yield (torch.randn(*sh, dtype=d, device=dev) * 2,)
        return make

    def silu2_inputs():
        def make(d):
            for sh in POINTWISE_SHAPES:
                x = torch.randn(*sh, dtype=d, device=dev) * 2
                yield (torch.cat([x, torch.randn_like(x)], dim=-1),)
        return make

    def bmm_inputs():
        def make(d):
            for (B, M, K, N) in BMM_SHAPES:
                yield (torch.randn(B, M, K, dtype=d, device=dev) * 0.3,
                       torch.randn(B, K, N, dtype=d, device=dev) * 0.3)
        return make

    def gelu_ref(x):
        return F.gelu(x.float(), approximate="tanh")

    def gam_ref(x, y):
        xf = x.float()
        inner = 0.7978845608028654 * (xf + 0.044715 * xf ** 3)
        return (0.5 * xf * (1 + torch.tanh(inner)) * y.float()).to(x.dtype)

    def sam_ref(xx):
        d = xx.shape[-1] // 2
        return (F.silu(xx[..., :d].float()) * xx[..., d:].float()).to(xx.dtype)

    rows = []

    def row(name, ours, fg, native, ref, inputs, relative=False):
        rows.append((name,
                     probe_grid(dev, ours, ref, inputs, DTYPES, relative),
                     probe_grid(dev, fg, ref, inputs, DTYPES, relative),
                     probe_grid(dev, native, ref, inputs, DTYPES, relative),
                     relative))

    row("gelu(tanh)",
        [lambda x: K1.gelu_tanh_triton(x)],
        [lambda x: ops.gelu(x, approximate="tanh")],
        [lambda x: F.gelu(x, approximate="tanh")],
        gelu_ref, pw_inputs())

    row("gelu_and_mul",
        [lambda x, y: RK.gelu_and_mul_triton(x, y)],
        [lambda x, y: ops.gelu(x) * y],
        [lambda x, y: F.gelu(x) * y],
        gam_ref, pw_inputs(two=True), relative=True)

    row("silu_and_mul",
        [lambda xx: RK.silu_and_mul_triton_counted(xx)],
        [lambda xx: ops.silu(xx[..., :xx.shape[-1] // 2])
                    * xx[..., xx.shape[-1] // 2:]],
        [lambda xx: F.silu(xx[..., :xx.shape[-1] // 2])
                    * xx[..., xx.shape[-1] // 2:]],
        sam_ref, silu2_inputs(), relative=True)

    row("softmax",
        [lambda x: smx_mod.softmax_triton(x)],
        [lambda x: ops.softmax(x, -1)],
        [lambda x: F.softmax(x, -1)],
        lambda x: F.softmax(x.float(), -1), pw_inputs())

    row("bmm",
        [lambda x, y: bmm_mod.bmm_triton(x, y)],
        [lambda x, y: ops.bmm(x, y)],
        [lambda x, y: torch.bmm(x, y)],
        lambda x, y: torch.bmm(x.float(), y.float()),
        bmm_inputs(), relative=True)

    print("| 算子 | dtype | 容差 | 自研 | FlagGems | 原生 |")
    print("|---|---|---|---|---|---|")
    for name, r, f, n, relative in rows:
        for dt in DTYPES:
            tol = 5e-2 if relative else (1e-5 if dt == "float32" else 1e-2)
            unit = "rel" if relative else "abs"
            print(f"| {name} | {dt} | {tol:.0e} {unit} | "
                  f"{fmt(r[dt], tol)} | {fmt(f[dt], tol)} | {fmt(n[dt], tol)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
