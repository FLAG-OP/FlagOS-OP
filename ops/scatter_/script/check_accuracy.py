"""Verify golden checksums and exact scatter results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from ops.scatter_ import scatter_reference, scatter_triton


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", required=True, help="cpu or device profile")
    parser.add_argument("--impl", choices=("torch", "triton"), required=True)
    parser.add_argument("--torch-device", help="Override profile's logical device")
    args = parser.parse_args()
    if args.device == "cpu":
        device = "cpu"
    else:
        from common.device import load_profile
        device = load_profile(args.device).torch_device
    if args.torch_device:
        device = args.torch_device
    fn = scatter_reference if args.impl == "torch" else scatter_triton
    root = Path(__file__).resolve().parents[1] / "goldendata/data"
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest:
        path = root / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        case = torch.load(path, weights_only=True)
        dst, index = case["dst"].to(device), case["index"].to(device)
        src = case["src"]
        if isinstance(src, torch.Tensor):
            src = src.to(device)
        result = fn(dst, case["dim"], index, src)
        assert result is dst
        torch.testing.assert_close(dst.cpu(), case["expected"], rtol=0, atol=0)
    print(f"PASS {len(manifest)} cases ({args.impl}, {device})")


if __name__ == "__main__":
    main()
