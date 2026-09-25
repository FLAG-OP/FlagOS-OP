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
        # 与 sdpa/_profile.py 同序: torch_mlu 优先，再 torch_xmlir，最后 cpu。
        try:
            import torch_mlu  # noqa: F401
            name = "cambricon"
        except ImportError:
            try:
                import torch_xmlir  # noqa: F401
                name = "p800-kunlunxin"
            except ImportError:
                name = "cpu"
    if name in ("p800-kunlunxin", "p800", "kunlunxin"):
        return LocalProfile(
            name="p800-kunlunxin",
            vendor="kunlunxin",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "cuda:1"),
            # Dense embedding autograd is intercepted on AutogradCUDA.
            dispatch_key="AutogradCUDA",
        )
    if name in ("cambricon", "mlu", "mlu590", "cambricon-mlu590"):
        # 注: 与 configs/devices/cambricon.yaml 的 PrivateUse1 不同——
        # embedding 是带 autograd 的 A1 交付，必须注册到 Autograd key 才能
        # 携带自带 backward（与 p800=AutogradCUDA、sdpa=AutogradPrivateUse1
        # 同构）。两个 key 实测都能命中，取 Autograd key 以免覆盖
        # functorch 的 PrivateUse1 batch rule（见 reports/cambricon.md §4）。
        return LocalProfile(
            name="cambricon",
            vendor="cambricon",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "mlu:0"),
            dispatch_key="AutogradPrivateUse1",
        )
    if name == "cpu":
        return LocalProfile("cpu", "cpu", "cpu", "CPU")
    raise ValueError(
        f"unknown embedding profile: {name}"
        "（内置 p800-kunlunxin / cambricon / cpu）"
    )
