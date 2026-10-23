#!/usr/bin/env python3
"""Run all matmul validation levels in isolated processes."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _run_command(command: list[str], label: str, timeout_s: int) -> None:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as cleanup_exc:
            raise RuntimeError(
                f"{label} exceeded {timeout_s}s; process {process.pid} did not exit"
            ) from cleanup_exc
        raise TimeoutError(f"{label} exceeded {timeout_s}s") from exc
    if code:
        raise subprocess.CalledProcessError(code, command)


def _run_level(level: str, profile_name: str, timeout_s: int) -> None:
    command = [sys.executable, str(OP_DIR / "test" / f"{level}_level.py"),
               profile_name]
    _run_command(command, f"{level} level", timeout_s)


def main() -> int:
    from common.device import load_profile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", nargs="?", default="p800-kunlunxin")
    ap.add_argument("--level-timeout", type=int, default=180)
    ap.add_argument("--probe-timeout", type=int, default=30)
    args = ap.parse_args()
    if args.level_timeout <= 0 or args.probe_timeout <= 0:
        ap.error("timeouts must be positive")
    profile = load_profile(args.profile)
    print(f"[matmul] {profile.summary()}", flush=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",")]
        if profile.torch_device.startswith("cuda:"):
            index = int(profile.torch_device.rsplit(":", 1)[1])
            if index < len(devices):
                print(f"[matmul] runtime CUDA_VISIBLE_DEVICES={visible}; "
                      f"{profile.torch_device} -> physical device {devices[index]}",
                      flush=True)
            else:
                ap.error(f"{profile.torch_device} is absent from CUDA_VISIBLE_DEVICES={visible}")
    try:
        if profile.vendor == "kunlunxin":
            probe = ("import sys, torch; x = torch.ones(1, device=sys.argv[1]); "
                     "print('device readback:', x.item(), flush=True)")
            print("[matmul] device readback probe", flush=True)
            _run_command([sys.executable, "-u", "-c", probe, profile.torch_device],
                         "device readback probe", args.probe_timeout)
        for level in ("kernel", "op", "framework"):
            print(f"\n[{level}]", flush=True)
            _run_level(level, profile.name, args.level_timeout)
    except TimeoutError as exc:
        print(f"[matmul] {exc}", file=sys.stderr)
        return 1
    print("\n[matmul] three levels passed in this environment", flush=True)
    return 0


def perf_cases(profile):
    from ops.matmul.script.perf_common import perf_cases as collect
    return collect(profile)


if __name__ == "__main__":
    raise SystemExit(main())
