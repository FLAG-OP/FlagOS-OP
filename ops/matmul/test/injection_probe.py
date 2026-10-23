#!/usr/bin/env python3
"""Check sitecustomize registration and marker-gated A1 hits in one process."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def main() -> None:
    boot = Path(os.environ["FLAGOS_MATMUL_BOOT_LOG"])
    hits = Path(os.environ["FLAGOS_MATMUL_HIT_LOG"])
    marker = Path(os.environ["FLAGOS_MATMUL_HIT_MARKER"])
    assert any("registered" in line for line in _lines(boot)), _lines(boot)
    assert not marker.exists(), f"marker already exists: {marker}"
    assert not _lines(hits), _lines(hits)

    from common.device import load_profile

    device = load_profile(os.environ["FLAGOS_MATMUL_PROFILE"]).torch_device
    a = torch.empty((0, 2), device=device)
    b = torch.empty((2, 2), device=device)
    torch.matmul(a, b)
    assert not _lines(hits), "A1 hit was logged before the marker"

    marker.touch()
    out = torch.matmul(a, b)
    assert tuple(out.shape) == (0, 2)
    assert len(_lines(hits)) == 1, _lines(hits)
    print(f"A1 injection probe passed: {device}, one marker-gated hit")


if __name__ == "__main__":
    main()
