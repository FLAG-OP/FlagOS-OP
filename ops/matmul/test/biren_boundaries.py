#!/usr/bin/env python3
"""Record Biren runtime support boundaries without masking failures."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
sys.path.insert(0, str(OP_DIR))
sys.path.insert(0, str(ROOT))


def _failure(exc):
    message = str(exc).strip().splitlines()[0]
    return f"{type(exc).__name__}: {message}"


def _probe_dtype(device, dtype):
    import torch

    left = torch.randn(63, 127).to(dtype).to(device)
    right = torch.randn(127, 65).to(dtype).to(device)
    out = torch.matmul(left, right)
    out.float().cpu()
    return "supported"


def _child(dtype_name):
    import torch

    from common.device import load_profile, load_runtime

    profile = load_profile("biren")
    load_runtime(profile)
    try:
        result = _probe_dtype(
            profile.torch_device, getattr(torch, dtype_name)
        )
    except Exception as exc:
        print(f"{dtype_name}={_failure(exc)}")
        return 1
    print(f"{dtype_name}={result}")
    return 0


def run(profile):
    import torch
    import torch_br  # noqa: F401
    import br_generator  # noqa: F401

    from common.device import load_runtime

    load_runtime(profile)
    device = profile.torch_device
    print(f"torch={torch.__version__}")
    print(f"torch_br={torch_br.__version__}")
    print(f"device={device} type={torch.empty(1, device=device).device.type}")

    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        try:
            results[str(dtype).split(".")[-1]] = _probe_dtype(device, dtype)
        except Exception as exc:  # Boundary probe must retain platform errors.
            results[str(dtype).split(".")[-1]] = _failure(exc)
    print(f"native_2d={results}")

    child = subprocess.run(
        [sys.executable, __file__, "--child-dtype", "float16"],
        text=True, capture_output=True,
    )
    details = [
        line for line in (child.stdout + child.stderr).splitlines()
        if "For br10x" in line or "RuntimeError" in line
    ]
    print(f"native_2d float16 child_exit={child.returncode}")
    for line in details[-3:]:
        print(f"  {line.strip()}")

    try:
        left = torch.randn(67).to(device)
        right = torch.randn(67).to(device)
        torch.matmul(left, right)
        results["native_vector_dot"] = "supported"
    except Exception as exc:
        results["native_vector_dot"] = _failure(exc)
    print(f"native_vector_dot={results['native_vector_dot']}")

    spec = importlib.util.find_spec("triton")
    if spec is None:
        results["triton"] = "not installed"
    else:
        triton = importlib.import_module("triton")
        results["triton"] = getattr(triton, "__version__", "unknown")
    print(f"triton={results['triton']}")

    spec = importlib.util.find_spec("flag_gems")
    if spec is None:
        results["flaggems"] = "not installed"
    else:
        flag_gems = importlib.import_module("flag_gems")
        version = getattr(flag_gems, "__version__", "unknown")
        try:
            flag_gems.enable()
        except Exception as exc:
            results["flaggems"] = f"{version}; enable failed: {_failure(exc)}"
        else:
            results["flaggems"] = f"{version}; enabled"
    print(f"flaggems={results['flaggems']}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child-dtype", choices=("float32", "float16", "bfloat16"))
    args = ap.parse_args()
    if args.child_dtype:
        raise SystemExit(_child(args.child_dtype))

    from common.device import load_profile
    print(run(load_profile("biren")))
