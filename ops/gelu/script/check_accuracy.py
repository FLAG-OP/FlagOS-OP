#!/usr/bin/env python3
"""gelu 黄金精度校验: 读 goldendata/data → 按容差判定（issue #55 要求 3）。

用法:
  python3 script/check_accuracy.py --impl triton --device cuda:6   # 自研（真机）
  python3 script/check_accuracy.py --impl native --device cuda:6   # 原生对照
  python3 script/check_accuracy.py --impl torch   --device cuda:6  # torch 级对照
  python3 script/check_accuracy.py --impl triton                   # CPU 自检
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

FLAGFIX_NEGATIVE_TAIL_X = [
    -6.076588153839111,
    -8.665115356445312,
    -8.432293891906738,
    -9.163774490356445,
    -9.992857933044434,
    -12.862730979919434,
    -7.933973789215088,
    -11.059422492980957,
]

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", choices=["triton", "native", "torch"],
                    default="triton")
    ap.add_argument("--device", default=None,
                    help="设备（如 cuda:6）; 缺省 CPU 自检")
    ap.add_argument("--flagfix-replay", action="store_true",
                    help="按 none/tanh × forward/backward 重放 FlagFix 负尾反例")
    args = ap.parse_args()

    ref_mod = _load("gelu_reference", "reference.py")
    if args.impl == "triton":
        impl_mod = _load("gelu_triton", "kernel/triton_level.py")
        fwd, bwd = impl_mod.gelu_triton, impl_mod.gelu_backward_triton
    elif args.impl == "native":
        impl_mod = _load("gelu_ppu", "kernel/thead_ppu.py")
        fwd, bwd = impl_mod.gelu_ppu, impl_mod.gelu_backward_ppu
    else:
        impl_mod = _load("gelu_torch", "kernel/torch_level.py")
        fwd, bwd = impl_mod.gelu_torch, None

    dev = torch.device(args.device) if args.device else torch.device("cpu")
    if args.device:
        torch.cuda.set_device(dev)

    if args.flagfix_replay:
        replay_passed = replay_failed = 0
        replay_x = torch.tensor(
            FLAGFIX_NEGATIVE_TAIL_X, dtype=torch.float32
        ).reshape(2, 4).to(dev)
        replay_grad = torch.ones_like(replay_x)
        for mode in ("none", "tanh"):
            reference_forward = ref_mod.gelu_reference(replay_x.cpu(), mode)
            reference_backward = ref_mod.gelu_backward_reference(
                replay_grad.cpu(), replay_x.cpu(), mode)
            checks = [("forward", fwd(replay_x, mode),
                       reference_forward.to(dev))]
            if bwd is not None:
                checks.append(("backward", bwd(replay_grad, replay_x, mode),
                               reference_backward.to(dev)))
            for operation, got, expected in checks:
                ok = torch.allclose(got, expected, rtol=1e-5, atol=1e-6)
                err = (got - expected).abs().max().item()
                print(f"FLAGFIX {mode}/{operation}: "
                      f"{'PASS' if ok else 'FAIL'} err={err:.9e}")
                replay_passed += ok
                replay_failed += not ok
        replay_total = replay_passed + replay_failed
        print(f"[gelu:{args.impl}@{dev}] flagfix "
              f"{replay_passed}/{replay_total} PASS")
        return 1 if replay_failed else 0

    index = json.loads(
        (HERE / "goldendata/data/index.json").read_text())
    passed, failed, observed, worst = 0, 0, 0, 0.0
    for entry in index["files"]:
        payload = torch.load(HERE / "goldendata/data" / entry["file"],
                             weights_only=False)
        x = payload["x"].to(dev)
        expected = payload["expected"].to(dev)
        mode = payload["mode"]
        if entry["tag"] == "backward":
            got = bwd(payload["grad"].to(dev), x, mode)
        else:
            got = fwd(x, mode)
        if entry["tag"] == "inf_info":
            # +inf 边界: 不定式平台分化，只记录、不判 PASS/FAIL
            print("INFO", entry["file"], "official=", expected.tolist())
            observed += 1
            continue
        if entry["tag"] == "nonfinite":
            # 位级 parity: nan↔nan / inf↔inf
            ok = (torch.equal(torch.isnan(got), torch.isnan(expected))
                  and torch.equal(
                      got[~torch.isnan(got.float())].float(),
                      expected[~torch.isnan(expected.float())].float()))
            err = 0.0 if ok else float("nan")
        else:
            tol = 1e-5 if entry["dtype"] == "float32" else 1e-2
            err = (got.float() - expected.float()).abs().max().item()
            ok = err < tol
        passed, failed = passed + ok, failed + (not ok)
        if ok and err == err:
            worst = max(worst, err)
        if not ok:
            print(f"FAIL {entry['file']} err={err}")
    total = passed + failed
    suffix = f", observed={observed}" if observed else ""
    print(f"[gelu:{args.impl}@{dev}] {passed}/{total} PASS, "
          f"worst={worst:.3e}{suffix}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
