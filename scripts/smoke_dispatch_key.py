#!/usr/bin/env python3
"""central profile dispatch_key 冒烟（PR36 条目 2: dispatch_key 分叉）。

验证中央 profile（configs/devices/<device>.yaml 的 `device.dispatch_key`）
在中央消费路径上「选得对、用得上」，三断言:

  1. 注册 key   —— 中央 harness 导出的 `A1_DISPATCH_KEY` 就是 YAML 声明值；
                   在与 injection/sitecustomize.py 相同的 env 契约下注册
                   `aten::silu` 后三上下文各真拦截 1 次（注册须持有
                   `torch.library.Library` 引用，见 known-issues #19.4）；同时
                   三个算子本地 `_profile` 仍为文档声明的 `AutogradCUDA`
                   （YAML 基键 / 算子 Autograd 键的分层不漂移）。
  2. 反向路径   —— 同一 key 注册下 requires_grad 前向带 grad_fn，backward
                   梯度有限且对齐 silu 解析反向。
  3. no_grad 一致 —— grad(无 requires_grad) / no_grad / inference_mode 三上下文
                   各拦截 1 次且输出逐位相同（基键口径; Autograd 键在
                   inference_mode 会零命中，见 docs/known-issues.md #19.3）。

用法:
  PYTHONPATH=<repo> python3 scripts/smoke_dispatch_key.py
  PYTHONPATH=<repo> python3 scripts/smoke_dispatch_key.py --device thead-ppu

口径: 按算子层惯例不改写 CUDA_VISIBLE_DEVICES（thead-ppu 的 16 卡直曝时
`cuda:6` = 物理卡 6，见 configs/devices/thead-ppu.yaml 注释）；跑卡前先
`ppu-smi` 确认 6/7 空闲。本脚本只断言、不写 results/，也不放宽任何既有测试。
退出码: 0=三断言全绿, 1=断言失败。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 中央注入 env 契约下的子进程探针（PYTHONPATH 含 injection/，与
# tests/framework_level 的子进程环境一致）
_CHILD = r"""
import json, os, sys, torch, torch.nn.functional as F
from routes.a1_aten import register_aten as RA

key = os.environ["A1_DISPATCH_KEY"]   # 与 injection/sitecustomize.py 同一 env 契约
dev = sys.argv[1]
lib = RA.register_silu_aten(key)      # 持有引用（known-issues #19.4；弃置即失效）
x = torch.randn(512, 512, device=dev)
out = {"key": key}
base = RA.CALL_COUNT["silu"]

y0 = F.silu(x)
out["grad"] = RA.CALL_COUNT["silu"] - base
base = RA.CALL_COUNT["silu"]
with torch.no_grad():
    y1 = F.silu(x)
out["no_grad"] = RA.CALL_COUNT["silu"] - base
base = RA.CALL_COUNT["silu"]
with torch.inference_mode():
    y2 = F.silu(x)
out["inference_mode"] = RA.CALL_COUNT["silu"] - base
out["equal"] = bool(torch.equal(y0, y1) and torch.equal(y0, y2))
out["total"] = RA.CALL_COUNT["silu"]
print("SMOKE_JSON " + json.dumps(out))
"""


def load_op_profile(op: str, name: str):
    """按文件路径独立加载 ops/<op>/_profile.py（三者同名模块，避免互相覆盖）。"""
    path = ROOT / "ops" / op / "_profile.py"
    spec = importlib.util.spec_from_file_location(f"smoke_{op}_profile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 处理需要模块已在 sys.modules
    spec.loader.exec_module(mod)
    return mod.load_profile(name)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  OK  {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="thead-ppu",
                    help="中央 profile 名（默认 thead-ppu）")
    args = ap.parse_args()

    from common.device import load_profile

    profile = load_profile(args.device)
    key = profile.dispatch_key
    dev = profile.torch_device
    print(f"central profile: {profile.summary()}")

    import torch
    import torch.nn.functional as F

    if dev.startswith("cuda:"):
        idx = int(dev.split(":", 1)[1])
        check(torch.cuda.is_available() and idx < torch.cuda.device_count(),
              f"设备可用: {dev} (device_count={torch.cuda.device_count()}, "
              "算子层口径不重映射 CUDA_VISIBLE_DEVICES)")

    # ---------- 断言 1: 注册 key ----------
    print("[1] 注册 key")
    from tests.framework_level import _harness

    exported = _harness._plugin_env("a1", profile)["A1_DISPATCH_KEY"]
    check(exported == key,
          f"中央 harness 导出 A1_DISPATCH_KEY == YAML dispatch_key ({key})")

    # 中央注入 env 契约（injection/sitecustomize.py 同款 PYTHONPATH 与变量）；
    # 注册按中央 op 层口径持有 Library 引用（tests/op_level/test_a1.py 同），
    # 计数即"key 选得对且真的拦到"的证据。
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'injection'}{os.pathsep}{ROOT}"
    env["FLAGOS_TEMPLATES_A1_INJECT"] = "1"
    env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
    env["A1_DISPATCH_KEY"] = key
    env.pop("A1_ATEN_COUNT_FILE", None)  # 冒烟不落计数文件
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, dev],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300)
    line = [l for l in child.stdout.splitlines() if l.startswith("SMOKE_JSON")]
    if not line:
        print(child.stdout, child.stderr)
        raise AssertionError("中央注入子进程未输出 SMOKE_JSON")
    got = json.loads(line[0].split("SMOKE_JSON ", 1)[1])
    check(got["key"] == key and got["grad"] == 1 and got["no_grad"] == 1
          and got["inference_mode"] == 1 and got["total"] == 3,
          f"中央 env 契约 A1_DISPATCH_KEY={key}: 三上下文各拦截 1 次 {got}")

    for op in ("sdpa", "embedding", "sdpa_math"):
        lp = load_op_profile(op, profile.name)
        check(lp.dispatch_key == "AutogradCUDA",
              f"ops/{op}/_profile.py 本地 key = AutogradCUDA"
              f"（YAML 基键分层不变, profile={profile.name}）")

    # ---------- 断言 2: 反向路径 ----------
    print("[2] 反向路径")
    from routes.a1_aten import register_aten as RA

    lib = RA.register_silu_aten(key)  # 必须持有引用（known-issues #19.4）
    g = torch.Generator(device="cpu").manual_seed(0)
    x0 = torch.randn(512, 512, generator=g).to(dev)

    RA.CALL_COUNT["silu"] = 0
    xr = x0.float().requires_grad_()
    yr = F.silu(xr)
    check(yr.grad_fn is not None and RA.CALL_COUNT["silu"] == 1,
          "requires_grad 前向带 grad_fn 且被拦截")
    yr.sum().backward()
    s = torch.sigmoid(xr.detach())
    ref = s * (1.0 + xr.detach() * (1.0 - s))
    err = (xr.grad - ref).abs().max().item()
    check(torch.isfinite(xr.grad).all() and err < 1e-5,
          f"backward 梯度有限且对齐解析解 (max_err={err:.3e})")

    # ---------- 断言 3: no_grad / inference 一致 ----------
    print("[3] no_grad / inference 一致")
    RA.CALL_COUNT["silu"] = 0
    y_grad = F.silu(x0)
    n_grad = RA.CALL_COUNT["silu"]
    with torch.no_grad():
        y_ng = F.silu(x0)
    n_ng = RA.CALL_COUNT["silu"]
    with torch.inference_mode():
        y_im = F.silu(x0)
    n_im = RA.CALL_COUNT["silu"]
    check(n_grad == 1 and n_ng == n_grad + 1 and n_im == n_ng + 1,
          f"grad/no_grad/inference_mode 各拦截 1 次 ({n_grad}/{n_ng}/{n_im})")
    check(torch.equal(y_grad, y_ng) and torch.equal(y_grad, y_im),
          "三上下文输出逐位相同")
    check(got["equal"], "子进程（中央 env 契约）口径与本进程结论一致")

    print("SMOKE PASS: 注册 key / 反向路径 / no_grad·inference 三断言全绿")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"SMOKE FAIL: {e}")
        raise SystemExit(1)
