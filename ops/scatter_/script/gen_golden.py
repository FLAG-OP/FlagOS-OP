"""Generate CPU golden tensors and SHA256 manifest (no accelerator access)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from ops.scatter_.test.cases import cases


def main():
    out = Path(__file__).resolve().parents[1] / "goldendata/data"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (name, dst, dim, index, src) in enumerate(cases()):
        expected = dst.clone().scatter_(dim, index, src)
        path = out / f"{i:03d}.pt"
        torch.save(dict(name=name, dst=dst, dim=dim, index=index, src=src,
                        expected=expected), path)
        manifest.append(dict(file=path.name, name=name,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(manifest)} CPU golden cases")


if __name__ == "__main__":
    main()
