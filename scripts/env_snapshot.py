#!/usr/bin/env python3
"""环境快照收集: 生成算子开发报告的「环境配置」章节。

自动收集: OS/硬件/软件栈版本/设备 profile/关键环境变量。
用法:
  python3 scripts/env_snapshot.py --device p800-kunlunxin            # 打印
  python3 scripts/env_snapshot.py --device ... --out env.md         # 落盘
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pkg_version(mod_name: str, attr: str = "__version__") -> str:
    try:
        mod = importlib.import_module(mod_name)
        return str(getattr(mod, attr, "unknown"))
    except Exception:
        return "(未安装)"


def _gpu_info(vendor: str) -> list[str]:
    """按厂商调用对应的 smi 工具（只读查询）。"""
    cmds = {
        "kunlunxin": ["xpu-smi"],
        "nvidia": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv"],
    }
    cmd = cmds.get(vendor)
    if not cmd:
        return [f"(厂商 {vendor} 无自动探测命令，请手填)"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = out.stdout.splitlines() if out.stdout else []
        # 保留表头 + 设备行（kunlunxin 输出较长，截取设备表）
        return lines[:24]
    except Exception:
        return [f"(执行 {' '.join(cmd)} 失败，请手填)"]


def collect(profile_name: str) -> str:
    from common.device import load_profile

    profile = load_profile(profile_name)

    sw_rows = [
        ("OS / 内核", f"{platform.system()} {platform.release()}"),
        ("Python", platform.python_version()),
        ("PyTorch", _pkg_version("torch")),
        ("vLLM", _pkg_version("vllm")),
        ("FlagGems", _pkg_version("flag_gems")),
        ("vllm-plugin-FL", _pkg_version("vllm_fl")),
        ("Triton", _pkg_version("triton")),
        ("transformers", _pkg_version("transformers")),
    ]

    env_keys = ["CUDA_VISIBLE_DEVICES", "VLLM_FL_PREFER", "VLLM_FL_PLATFORM",
                "USE_FLAGGEMS", "VLLM_FL_PER_OP", "VLLM_FL_PLUGIN_MODULES"]
    env_rows = [(k, os.environ.get(k, "(未设置)")) for k in env_keys]

    gpu = "\n".join("    " + l for l in _gpu_info(profile.vendor))

    lines = [
        "## 1. 环境配置",
        "",
        f"> 自动生成: `scripts/env_snapshot.py --device {profile_name}`；"
        "标注 `[手填]` 的项需人工补充。",
        "",
        "### 1.1 硬件",
        "",
        f"设备 profile: **{profile.name}**（vendor={profile.vendor}，"
        f"torch_device={profile.torch_device}，visible={profile.visible_devices}）",
        "",
        "```",
        gpu,
        "```",
        "",
        "拓扑 / 卡号映射 `[手填]`",
        "",
        "### 1.2 软件栈",
        "",
        "| 组件 | 版本 |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in sw_rows] + [
        "| 厂商库 | [手填] |",
        "",
        "### 1.3 设备 profile",
        "",
        "```yaml",
        (ROOT / "configs/devices" / f"{profile_name}.yaml").read_text().rstrip(),
        "```",
        "",
        "### 1.4 关键环境变量（采集时）",
        "",
        "| 变量 | 值 |",
        "|---|---|",
    ] + [f"| `{k}` | `{v}` |" for k, v in env_rows] + [
        "",
        "### 1.5 环境特殊性说明 `[手填]`",
        "",
        "<容器限制 / 共享设备 / quirks 开关原因>",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = collect(args.device)
    if args.out:
        Path(args.out).write_text(text)
        print(f"已写入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
