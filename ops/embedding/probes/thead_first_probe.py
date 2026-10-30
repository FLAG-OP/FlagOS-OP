#!/usr/bin/env python3
"""thead-ppu embedding 首测探针（任务书 §2b）。

覆盖:
  A. aten::embedding / index_select / embedding_backward overload 存在性
  B. native embedding vs CPU 参考精度（3 dtype × padding 组合）
  C. embedding_backward 的 scale_grad_by_freq / padding_idx 厂商边界
     （cambricon 先例: 原生完整; p800 先例: scale 抛 ValueError）
  D. dispatch key 命中计数: AutogradCUDA vs CUDA
     （call count>0 才算命中——Ascend 教训; 计数 wrapper 直调 index_select
      委托面, 不回 F.embedding 防递归; 每 key 独立子进程, #19.4）

运行:
  python3 probes/thead_first_probe.py            # A+B+C（单进程）
  python3 probes/thead_first_probe.py --key AutogradCUDA
  python3 probes/thead_first_probe.py --key CUDA
JSON: probes/out/thead_first_{overloads,key_<KEY>}.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402

DEV = "cuda:6"
CALLS = {"n": 0}


def _counting_embedding(weight, indices, padding_idx=-1,
                        scale_grad_by_freq=False, sparse=False):
    """计数 wrapper: 命中后走 index_select 委托面（不回 F.embedding 防递归）。"""
    CALLS["n"] += 1
    if indices.dim() == 1:
        return torch.ops.aten.index_select(weight, 0, indices)
    flat = indices.reshape(-1)
    out = torch.ops.aten.index_select(weight, 0, flat)
    return out.view(*indices.shape, weight.shape[-1])


def probe_overloads() -> dict:
    report: dict = {"device": DEV}
    report["aten_embedding_has_schema"] = hasattr(torch.ops.aten, "embedding")
    report["aten_index_select_has_schema"] = hasattr(
        torch.ops.aten, "index_select"
    )
    report["aten_embedding_backward_has_schema"] = hasattr(
        torch.ops.aten, "embedding_backward"
    )
    report["device_name"] = torch.cuda.get_device_name(torch.device(DEV))
    report["is_ppu"] = "PPU" in report["device_name"]

    # B) native embedding 精度 vs CPU 官方（含 padding_idx）
    torch.manual_seed(0)
    fwd = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for pad in (-1, 3):
            num_weights, dim = 64, 16
            weight = (torch.randn(num_weights, dim) * 0.1).to(dtype)
            indices = torch.randint(0, num_weights, (3, 17))
            nat = torch.ops.aten.embedding(
                weight.to(DEV), indices.to(DEV), pad, False, False
            )
            ref = torch.nn.functional.embedding(
                indices, weight, padding_idx=pad
            )
            err = (nat.cpu().float() - ref.float()).abs().max().item()
            fwd.append(
                {"dtype": str(dtype).split(".")[-1], "pad": pad, "err": err}
            )
    report["native_forward"] = fwd

    # index_select 委托面精度
    weight = (torch.randn(64, 16) * 0.1).to(torch.float16).to(DEV)
    idx = torch.randint(0, 64, (128,), device=DEV)
    got = torch.ops.aten.index_select(weight, 0, idx).cpu()
    exp = torch.index_select(weight.cpu(), 0, idx.cpu())
    report["index_select_err"] = (
        got.float() - exp.float()
    ).abs().max().item()

    # C) embedding_backward 厂商边界: scale_grad_by_freq × padding_idx
    bwd = []
    num_weights = 16
    grad_out = (torch.randn(2, 3, 12) * 0.1).to(torch.float16).to(DEV)
    indices = torch.tensor([[0, 1, 1], [3, 3, 5]], device=DEV)
    for scale in (False, True):
        for pad in (-1, 3):
            try:
                out = torch.ops.aten.embedding_backward(
                    grad_out, indices, num_weights, pad,
                    scale_grad_by_freq=scale, sparse=False,
                )
                w = torch.randn(num_weights, 12, requires_grad=True)
                g = grad_out.cpu().float()
                torch.nn.functional.embedding(
                    indices.cpu(), w, padding_idx=pad,
                    scale_grad_by_freq=scale,
                ).backward(g)
                err = (out.float().cpu() - w.grad.float()).abs().max().item()
                bwd.append(
                    {"scale": scale, "pad": pad, "status": "ok", "err": err}
                )
            except Exception as e:  # noqa: BLE001 探针记录任意厂商报错
                bwd.append(
                    {"scale": scale, "pad": pad, "status": "error",
                     "error": f"{type(e).__name__}: {e}"}
                )
    report["native_backward"] = bwd
    return report


def probe_key(key: str) -> dict:
    """D) dispatch key 命中（T1 thead_reg_probe.py 同款: 计数 wrapper 直注册）。"""
    import torch.nn.functional as F

    res: dict = {"dispatch_key": key, "device": DEV, "probes": {}}
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("embedding", _counting_embedding, key)
    res["has_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::embedding", key)

    g = torch.Generator(device="cpu").manual_seed(3)

    def mk(requires_grad):
        w = (
            (torch.randn(32, 8, generator=g) * 0.1)
            .to(torch.float16).to(DEV).requires_grad_(requires_grad)
        )
        idx = torch.tensor([[0, 1, 1], [3, 3, 31]], device=DEV)
        return w, idx

    # 1) grad mode, no requires_grad（kernel/op 层测试口径）
    w, idx = mk(False)
    CALLS["n"] = 0
    _ = F.embedding(idx, w, padding_idx=3)
    res["probes"]["grad_no_rg"] = CALLS["n"]

    # 2) grad mode + requires_grad + backward
    w, idx = mk(True)
    CALLS["n"] = 0
    out = F.embedding(idx, w, padding_idx=3)
    res["probes"]["grad_rg_has_grad_fn"] = out.grad_fn is not None
    try:
        out.float().sum().backward()
        bwd_ok = bool(w.grad is not None and torch.isfinite(w.grad).all())
    except Exception as e:  # noqa: BLE001
        bwd_ok = f"{type(e).__name__}: {str(e)[:120]}"
    res["probes"]["grad_rg"] = CALLS["n"]
    res["probes"]["grad_rg_backward"] = bwd_ok

    # 3) torch.no_grad（framework 层测试口径）
    w, idx = mk(False)
    CALLS["n"] = 0
    with torch.no_grad():
        _ = F.embedding(idx, w, padding_idx=3)
    res["probes"]["no_grad"] = CALLS["n"]

    # 4) inference_mode（known-issues #19 教训: 可能零命中）
    w, idx = mk(False)
    CALLS["n"] = 0
    with torch.inference_mode():
        _ = F.embedding(idx, w, padding_idx=3)
    res["probes"]["inference_mode"] = CALLS["n"]

    # 5) 直调 aten（绕 F 包装）
    w, idx = mk(False)
    CALLS["n"] = 0
    _ = torch.ops.aten.embedding.default(w, idx, 3, False, False)
    res["probes"]["aten_direct"] = CALLS["n"]

    hits = [k for k, v_ in res["probes"].items()
            if isinstance(v_, int) and v_ > 0]
    res["hit_probes"] = hits
    print(f"[{key}] has_kernel={res['has_kernel']} probes={res['probes']}")
    print(f"[{key}] HIT: {hits or 'NONE'}")
    return res


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--key":
        key = sys.argv[2]
        res = probe_key(key)
        out = OP_DIR / "probes" / "out" / f"thead_first_key_{key}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("JSON:", out)
        return 0 if res["hit_probes"] else 1

    res = probe_overloads()
    out = OP_DIR / "probes" / "out" / "thead_first_overloads.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
