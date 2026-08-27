#!/usr/bin/env python3
"""按环境锁定文件核对当前容器（共享镜像自检）。

用法:
  python3 scripts/check_env.py                          # 默认 p800-kunlunxin
  python3 scripts/check_env.py --require-model          # 顺带检查模型路径

模型路径解析优先级: 环境变量 FLAGOS_MODEL_PATH > 设备 profile。
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "configs" / "env"

# 模块名 → 发行包名（importlib.metadata 用发行包名）
DIST = {"flag_gems": "flag-gems", "vllm_fl": "vllm-plugin-fl"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--require-model", action="store_true",
                    help="检查框架级模型路径是否存在")
    args = ap.parse_args()

    lock_path = ENV_DIR / f"{args.device}.lock.yaml"
    if not lock_path.is_file():
        print(f"环境锁定文件不存在: {lock_path}")
        return 2
    lock = yaml.safe_load(lock_path.read_text())

    print(f"[check-env] {lock['description']} (python {lock['python']})")
    print(f"{'包':10s} {'锁定':24s} {'当前':24s} 结果")
    print("-" * 66)

    n_bad = 0
    for pkg, want in lock["packages"].items():
        dist = DIST.get(pkg, pkg)
        try:
            got = md.version(dist)
        except md.PackageNotFoundError:
            got = "(未安装)"
        ok = got == want
        n_bad += not ok
        print(f"{pkg:10s} {want:24s} {got:24s} {'✓' if ok else '✗'}")

    py = platform.python_version()
    py_ok = py == lock["python"]
    n_bad += not py_ok
    print(f"{'python':10s} {lock['python']:24s} {py:24s} {'✓' if py_ok else '✗'}")

    if lock.get("notes"):
        print(f"\n说明: {lock['notes'].strip()}")

    if args.require_model:
        sys.path.insert(0, str(ROOT))
        from common.device import load_profile
        profile = load_profile(args.device)
        model = profile.engine_args().get("model")
        exists = Path(model).is_dir() if model else False
        n_bad += not exists
        src = "环境变量" if os.environ.get("FLAGOS_MODEL_PATH") else "profile"
        print(f"\n模型路径[{src}]: {model} {'✓' if exists else '✗ 不存在'}")

    print(f"\n结论: {'OK' if n_bad == 0 else f'{n_bad} 项不符'}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
