#!/usr/bin/env python3
"""Guard behavior check: metadata, happy path, CPU rejection, registration."""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))

import torch  # noqa: E402

from _profile import load_profile  # noqa: E402
from kernel.backends import get_impl  # noqa: E402
from kernel.triton_level import sdpa_triton  # noqa: E402


def main() -> int:
    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    device_type = profile.torch_device.split(":", 1)[0]
    if device_type == "npu":
        import torch_npu  # noqa: F401
    if device_type == "mlu":
        import torch_mlu  # noqa: F401

    backend = get_impl(device_type)
    assert backend.PLATFORM == profile.name
    assert device_type in backend.SUPPORTED_DEVICE_TYPES
    print(f"metadata OK: PLATFORM={backend.PLATFORM} "
          f"types={backend.SUPPORTED_DEVICE_TYPES}")

    g = torch.Generator(device="cpu").manual_seed(1)
    q = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16)
    k = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16)
    v = (torch.randn(1, 4, 64, 64, generator=g) * 0.5).to(torch.float16)
    q, k, v = (t.to(profile.torch_device) for t in (q, k, v))
    out = sdpa_triton(q, k, v, None, 0.0, True, None, False)
    print(f"{profile.torch_device} path OK: {tuple(out.shape)}")

    qc, kc, vc = q.cpu(), k.cpu(), v.cpu()
    try:
        sdpa_triton(qc, kc, vc, None, 0.0, True, None, False)
    except RuntimeError as e:
        assert ("无 'cpu' 平台实现" in str(e)
                or backend.PLATFORM in str(e)) and "PLATFORM.md" in str(e)
        print("CPU rejection OK:", str(e)[:72], "...")
    else:
        raise AssertionError("CPU call was not rejected")

    from register import register_a1
    lib = register_a1(profile.dispatch_key)
    print(f"registration guard OK: {profile.dispatch_key} can register")
    del lib
    print("ALL GUARD CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
