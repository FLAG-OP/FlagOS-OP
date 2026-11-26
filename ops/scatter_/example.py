"""Run the per-op three-layer suite only on an explicitly selected profile."""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("profile")
    parser.add_argument("--torch-device", help="Logical device within visible devices")
    parser.add_argument("--out", type=Path, help="Write structured test results")
    args = parser.parse_args()
    from common.device import load_profile
    profile = load_profile(args.profile)
    if args.torch_device:
        profile.torch_device = args.torch_device
    results = {"profile": profile.name, "torch_device": profile.torch_device}
    for level in ("kernel", "op", "framework"):
        result = importlib.import_module(f"ops.scatter_.test.{level}_level").run(profile)
        if not result["ok"]:
            raise RuntimeError(f"{level} failed: {result}")
        print(level, result)
        results[level] = result
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")


def perf_cases(profile):
    from common.perf import PerfCase
    from ops.scatter_ import scatter_reference, scatter_triton

    def make(fn, shape):
        def build(p):
            import torch
            dst = torch.zeros(shape, device=p.torch_device)
            src = torch.ones_like(dst)
            idx = torch.arange(shape[-1], device=dst.device).flip(0).expand(shape)
            return lambda: fn(dst, -1, idx, src)
        return build

    return [PerfCase(f"ops.scatter_.{m}.{rows}x{cols}", group="ops", level="kernel",
                     make_fn=make(fn, (rows, cols)))
            for rows, cols in [(7, 1031), (128, 4096)]
            for m, fn in [("triton", scatter_triton), ("torch", scatter_reference)]]


if __name__ == "__main__":
    main()
