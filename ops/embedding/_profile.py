"""Local profile shim for standalone operator execution."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LocalProfile:
    name: str
    vendor: str
    torch_device: str
    dispatch_key: str

    def summary(self) -> str:
        return (f"{self.name} [vendor={self.vendor} "
                f"dev={self.torch_device} key={self.dispatch_key}]")


def load_profile(name: str | None = None) -> LocalProfile:
    name = name or os.environ.get("EMBEDDING_PROFILE")
    if name is None:
        try:
            import torch_xmlir  # noqa: F401
            name = "p800-kunlunxin"
        except ImportError:
            name = "cpu"
    if name == "p800-kunlunxin":
        return LocalProfile(
            name=name,
            vendor="kunlunxin",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "cuda:1"),
            # Dense embedding autograd is intercepted on AutogradCUDA.
            dispatch_key="AutogradCUDA",
        )
    if name == "cpu":
        return LocalProfile("cpu", "cpu", "cpu", "CPU")
    raise ValueError(f"unknown embedding profile: {name}")
