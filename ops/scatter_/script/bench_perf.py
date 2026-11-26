"""Synchronized native/Triton comparison using the repository timing API."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--torch-device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from common.device import load_profile
    from common.perf import run_case
    from ops.scatter_.example import perf_cases
    profile = load_profile(args.device)
    profile.torch_device = args.torch_device
    records = [run_case(case, profile).to_dict() for case in perf_cases(profile)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    for row in records:
        print(row["case_id"], row["metrics"])


if __name__ == "__main__":
    main()
