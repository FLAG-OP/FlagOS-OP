#!/usr/bin/env python3
"""thead-ppu 稠密组 P0 探针（issue #55/#56/#57 前置取证）。

在真武 810E（默认 cuda:6，跑前 ppu-smi 确认空闲）上逐项探测:
  - issue #55 七算子的原生 aten 可用性与精度（fp16/bf16/fp32）
  - issue #56 aten::_efficient_attention_forward 直调可用性
  - issue #57 CSR 稀疏 mm / _sparse_mm 可用性

用法:
  python3 probes/thead_dense_p0_probe.py                # 顺序跑全部
  python3 probes/thead_dense_p0_probe.py --item linear  # 单项（子进程隔离,
      某项触发栈级 C++ abort 时不影响其余项; 与 --all 组合使用见 run_all()）
  python3 probes/thead_dense_p0_probe.py --isolated     # 每项独立子进程

输出: probes/out/thead_dense_p0_probe.json（各项结果合并）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "probes" / "out" / "thead_dense_p0_probe.json"
ITEMS = ["linear", "layer_norm", "native_layer_norm", "matmul", "softmax",
         "_softmax", "gelu", "gelu_backward", "efficient_attention",
         "sparse_mm"]


def maxerr(a, b):
    return (a.float() - b.float()).abs().max().item()


def ref_gelu(x: torch.Tensor, approx: str) -> torch.Tensor:
    xf = x.double()
    if approx == "none":
        r = xf * 0.5 * (1.0 + torch.erf(xf / math.sqrt(2.0)))
    else:
        r = 0.5 * xf * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf ** 3)))
    return r.to(x.dtype)


def probe_linear(dev):
    out = {}
    x = torch.randn(8, 64, dtype=torch.float16, device=dev)
    w = torch.randn(32, 64, dtype=torch.float16, device=dev)
    b = torch.randn(32, dtype=torch.float16, device=dev)
    got = torch.ops.aten.linear(x, w, b)
    ref = torch.nn.functional.linear(x.cpu().float(), w.cpu().float(),
                                     b.cpu().float()).to(dev)
    out["available"] = True
    out["fp16_err"] = maxerr(got, ref.to(got.dtype))
    x3 = torch.randn(4, 8, 64, dtype=torch.bfloat16, device=dev)
    w3 = w.to(torch.bfloat16)
    got3 = torch.ops.aten.linear(x3, w3, None)
    ref3 = torch.nn.functional.linear(
        x3.cpu().float(), w3.cpu().float(), None).to(dev)
    out["bf16_3d_err"] = maxerr(got3, ref3.to(got3.dtype))
    return out


def probe_layer_norm(dev, native=False):
    out = {}
    x = torch.randn(16, 128, dtype=torch.float16, device=dev)
    w = torch.randn(128, dtype=torch.float16, device=dev)
    b = torch.randn(128, dtype=torch.float16, device=dev)
    if native:
        res = torch.ops.aten.native_layer_norm(x, [128], w, b, 1e-5)
        got = res[0]
        out["n_returns"] = len(res)
    else:
        got = torch.ops.aten.layer_norm(x, [128], w, b, 1e-5, False)
    ref = torch.nn.functional.layer_norm(
        x.cpu().float(), (128,), w.cpu().float(), b.cpu().float(),
        1e-5).to(dev)
    out["available"] = True
    out["fp16_err"] = maxerr(got, ref.to(got.dtype))
    return out


def probe_matmul(dev):
    out = {}
    a = torch.randn(64, 128, dtype=torch.float16, device=dev)
    b2 = torch.randn(128, 96, dtype=torch.float16, device=dev)
    got = torch.ops.aten.matmul(a, b2)
    ref = (a.cpu().float() @ b2.cpu().float()).to(dev)
    out["available"] = True
    out["fp16_err"] = maxerr(got, ref.to(got.dtype))
    a3 = torch.randn(4, 64, 128, dtype=torch.bfloat16, device=dev)
    b3 = b2.to(torch.bfloat16)
    got3 = torch.ops.aten.matmul(a3, b3)
    ref3 = (a3.cpu().float() @ b3.cpu().float()).to(dev)
    out["bf16_broadcast_err"] = maxerr(got3, ref3.to(got3.dtype))
    return out


def probe_softmax(dev, private=False):
    out = {}
    x = torch.randn(32, 512, dtype=torch.float16, device=dev) * 8
    ref = torch.softmax(x.cpu().float(), -1).to(dev)
    if private:
        got = torch.ops.aten._softmax(x, -1, False)
        out["available"] = True
        out["fp16_err"] = maxerr(got, ref.to(got.dtype))
        g2 = torch.ops.aten._softmax(x, -1, True)
        out["halves_to_float_ok"] = True
        out["out_dtype"] = str(g2.dtype)
        out["h2f_err"] = maxerr(
            g2, torch.softmax(x.cpu().float(), -1).to(dev))
        return out
    got = None
    # 公开 API 先行（production 路由依据）; aten::softmax 3 参直调的
    # host_softmax Byte 怪癖单列取证（首轮真机事实）。
    for name, fn in [
        ("F_softmax", lambda: torch.nn.functional.softmax(x, -1)),
        ("Tensor_softmax", lambda: x.softmax(-1)),
        ("torch_softmax", lambda: torch.softmax(x, -1)),
        ("aten_softmax_3arg", lambda: torch.ops.aten.softmax(x, -1, False)),
    ]:
        try:
            g = fn()
            out[name] = {"ok": True,
                         "err": maxerr(g, ref.to(g.dtype))}
            if got is None:
                got = g
        except Exception as e:
            out[name] = {"ok": False, "error": repr(e)[:200]}
    out["available"] = got is not None
    if got is not None:
        out["fp16_err"] = out.get("F_softmax", {}).get(
            "err", maxerr(got, ref.to(got.dtype)))
    return out


def probe_gelu(dev):
    # schema 实测: approximate 是 keyword-only（首版位置传参 schema 不匹配）
    out = {}
    for approx in ("none", "tanh"):
        x = torch.randn(4096, dtype=torch.float16, device=dev) * 3
        got = torch.ops.aten.gelu(x, approximate=approx)
        ref = ref_gelu(x.cpu(), approx).to(dev)
        out[f"{approx}_available"] = True
        out[f"{approx}_fp16_err"] = maxerr(got, ref.to(got.dtype))
    return out


def probe_gelu_backward(dev):
    out = {}
    x = (torch.randn(1024, dtype=torch.float16, device=dev) * 3
         ).requires_grad_(True)
    y = torch.ops.aten.gelu(x, approximate="none")
    (gx,) = torch.autograd.grad(y.sum(), x)
    xf = x.detach().cpu().double()
    d = 0.5 * (1 + torch.erf(xf / math.sqrt(2))) + \
        xf * torch.exp(-xf * xf / 2) / math.sqrt(2 * math.pi)
    out["available"] = True
    out["fp16_err"] = maxerr(gx, d.to(gx.device).to(gx.dtype))
    return out


def probe_efficient_attention(dev):
    out = {}
    try:
        op = torch.ops.aten._efficient_attention_forward
        out["op_exists"] = True
        out["schema"] = str(op.default._schema)
    except Exception as e:
        out["op_exists"] = False
        out["error"] = repr(e)[:300]
        return out
    B, H, S, D = 2, 4, 64, 32
    q = torch.randn(B, H, S, D, dtype=torch.float16, device=dev)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=dev)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=dev)
    # schema: (q, k, v, bias, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    #          max_seqlen_k, dropout_p, custom_mask_type,
    #          compute_log_sumexp=False, *, scale, seqlen_k, window_size)
    # bias 形状逐候选取证（首轮 (B,H,S,S) 报 wrong shape (head dimension)）:
    #   None / (1,1,S,S) / (B,1,S,S) / (B,H,S,S)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.cpu().float(), k.cpu().float(), v.cpu().float()).to(dev)
    for tag, bias in [
        ("bias_none", None),
        ("bias_1_1", torch.zeros(1, 1, S, S, dtype=torch.float16, device=dev)),
        ("bias_b_1", torch.zeros(B, 1, S, S, dtype=torch.float16, device=dev)),
        ("bias_b_h", torch.zeros(B, H, S, S, dtype=torch.float16, device=dev)),
    ]:
        try:
            # scale=None → 沿用 F.sdpa 默认 1/sqrt(D)（首轮 scale=1.0 与
            # 参考口径不一致造成 3.678 假误差, 已修正）
            res = torch.ops.aten._efficient_attention_forward(
                q, k, v, bias, None, None, None, None, 0.0, 0, True)
            o, lse = res[0], res[1]
            entry = {"ok": True, "n_returns": len(res),
                     "out_shape": list(o.shape),
                     "lse_shape": list(lse.shape),
                     "philox_seed_dtype": str(res[2].dtype),
                     "max_seqlen": [str(res[4]), str(res[5])]}
            # 布局/缩放判别: 输出按 BHSD 直接比 + 转置比 × scale 两口径
            o_f = o.float()
            entry["err_bhsd_default"] = maxerr(o, ref)
            entry["err_bhsd_scale1"] = maxerr(
                o, torch.nn.functional.scaled_dot_product_attention(
                    q.cpu().float(), k.cpu().float(), v.cpu().float(),
                    scale=1.0).to(dev))
            # BSHD-in-BHSD 假设: kernel 按 BSHD 写内存但张量形状是 BHSD →
            # 内存重解释 reshape(B,S,H,D).transpose(1,2) 后应与参考一致
            if o.dim() == 4 and o.shape == tuple(ref.shape):
                entry["err_bshd_reinterp"] = maxerr(
                    o.reshape(B, S, H, D).transpose(1, 2), ref)
                entry["lse_as_bsh_pad"] = [
                    list(lse.reshape(B, S, -1).transpose(1, 2).shape),
                    float(lse.reshape(B, S, -1).transpose(1, 2)[..., :H]
                          .float().abs().max())]
            out[tag] = entry
        except Exception as e:
            out[tag] = {"ok": False, "error": repr(e)[:250]}
    # 变体矩阵: compute_log_sumexp / 显式 scale / dtype / D——收口
    # "最小失败输入"证据（issue #56: 列出失败的最小输入、预期与实际）。
    import math as _math
    variants = {}
    for vtag, kw in [
        ("no_lse", dict(cls=False)),
        ("scale_explicit", dict(scale=1.0 / _math.sqrt(D))),
        ("fp32", dict(dt=torch.float32)),
        ("d64", dict(D=64)),
        ("s128", dict(S=128)),
    ]:
        try:
            dt_v = kw.pop("dt", torch.float16)
            D_v = kw.pop("D", D)
            S_v = kw.pop("S", S)
            qv = torch.randn(B, H, S_v, D_v, dtype=dt_v, device=dev)
            kv = torch.randn(B, H, S_v, D_v, dtype=dt_v, device=dev)
            vv = torch.randn(B, H, S_v, D_v, dtype=dt_v, device=dev)
            scale_v = kw.pop("scale", None)
            r_v = torch.nn.functional.scaled_dot_product_attention(
                qv.cpu().float(), kv.cpu().float(), vv.cpu().float(),
                scale=scale_v).to(dev)
            res_v = torch.ops.aten._efficient_attention_forward(
                qv, kv, vv, None, None, None, None, None, 0.0, 0,
                **({"compute_log_sumexp": False} if not kw.pop("cls", True)
                   else {}), scale=scale_v)
            variants[vtag] = {
                "ok": True, "out_shape": list(res_v[0].shape),
                "err_bhsd_default": maxerr(res_v[0], r_v),
                "lse_shape": list(res_v[1].shape)}
        except Exception as e:
            variants[vtag] = {"ok": False, "error": repr(e)[:220]}
    out["variants"] = variants
    return out


def probe_sparse_mm(dev, dt_name="fp32", private=False):
    """CSR × dense 逐 dtype × 逐入口取证，含冷/暖双调（首发 lazy-init 假设）。

    首轮真机事实: torch.sparse.mm fp32 在暖进程成功（err 8.6e-6）、
    冷进程首发报 cusparseSpMM_bufferSize not implemented; fp16 触发
    同报错。本节每 dtype × 每入口连续调两次, 区分冷/暖行为。
    """
    dt = {"fp32": torch.float32, "fp16": torch.float16,
          "bf16": torch.bfloat16}[dt_name]
    M, K, N = 64, 32, 16
    dense = torch.randn(M, K, dtype=dt, device=dev)
    dense[dense.abs() < 0.7] = 0
    csr = dense.to_sparse_csr()
    rhs = torch.randn(K, N, dtype=dt, device=dev)
    ref = dense @ rhs
    out = {}
    for entry, fn in [("sparse_mm", lambda: torch.sparse.mm(csr, rhs)),
                      ("_sparse_mm", lambda: torch.ops.aten._sparse_mm(csr, rhs))]:
        calls = []
        for i in range(2):   # 第 1 次=冷, 第 2 次=暖（lazy-init 取证）
            try:
                got = fn()
                calls.append({"ok": True, "err": maxerr(got, ref),
                              "out_shape": list(got.shape)})
            except Exception as e:
                calls.append({"ok": False, "error": repr(e)[:220]})
        out[entry] = {"cold": calls[0], "warm": calls[1]}
    return out


PROBES = {
    "linear": lambda dev: probe_linear(dev),
    "layer_norm": lambda dev: probe_layer_norm(dev),
    "native_layer_norm": lambda dev: probe_layer_norm(dev, native=True),
    "matmul": lambda dev: probe_matmul(dev),
    "softmax": lambda dev: probe_softmax(dev),
    "_softmax": lambda dev: probe_softmax(dev, private=True),
    "gelu": lambda dev: probe_gelu(dev),
    "gelu_backward": lambda dev: probe_gelu_backward(dev),
    "efficient_attention": lambda dev: probe_efficient_attention(dev),
    "sparse_mm": lambda dev: {k: probe_sparse_mm(dev, k)
                              for k in ("fp32", "fp16", "bf16")},
}


def run_items(device, items):
    dev = torch.device(device)
    results = {"device": device,
               "device_name": torch.cuda.get_device_name(dev),
               "torch": torch.__version__, "items": {}}
    for item in items:
        try:
            results["items"][item] = PROBES[item](dev)
        except Exception as e:
            results["items"][item] = {"error": repr(e)[:400]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


def run_isolated(device, items):
    """每项独立子进程——栈级 C++ abort 只损失该项（隔离取证）。"""
    results = {"device": device, "items": {}}
    for item in items:
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--device", device, "--item", item]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=120, cwd=str(ROOT))
            if proc.returncode == 0:
                results["items"][item] = json.loads(proc.stdout)
            else:
                tail = (proc.stderr or "").strip().splitlines()[-3:]
                results["items"][item] = {
                    "crashed": True, "returncode": proc.returncode,
                    "stderr_tail": tail}
        except subprocess.TimeoutExpired:
            results["items"][item] = {"crashed": True, "timeout": 120}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("PPU_PROBE_DEVICE",
                                                       "cuda:6"))
    ap.add_argument("--item", choices=ITEMS, default=None)
    ap.add_argument("--isolated", action="store_true",
                    help="每项独立子进程（默认: 同进程顺序执行）")
    args = ap.parse_args()
    items = [args.item] if args.item else ITEMS
    if args.isolated and args.item is None:
        run_isolated(args.device, items)
    else:
        run_items(args.device, items)


if __name__ == "__main__":
    main()
