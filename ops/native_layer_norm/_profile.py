"""Local profile shim for standalone native_layer_norm execution."""
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


def _is_ppu_stack() -> bool:
    try:
        from triton.runtime import driver
        if "PPU" in type(driver.active).__name__:
            return True
    except Exception:
        pass
    try:
        import torch
        return torch.cuda.is_available() and any(
            "PPU" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


def load_profile(name: str | None = None) -> LocalProfile:
    name = name or os.environ.get("NATIVE_LAYER_NORM_PROFILE")
    if name is None:
        try:
            import torch_mlu  # noqa: F401
            name = "cambricon"
        except ImportError:
            try:
                import torch_xmlir  # noqa: F401
                name = "p800-kunlunxin"
            except ImportError:
                if _is_ppu_stack():
                    name = "thead-ppu"
                else:
                    try:
                        import torch_npu  # noqa: F401
                        name = "ascend910"
                    except ImportError:
                        name = "cpu"
    if name in ("thead-ppu", "thead", "ppu", "thead-ppu-zw810e"):
        return LocalProfile(
            name="thead-ppu",
            vendor="thead",
            torch_device=os.environ.get(
                "NATIVE_LAYER_NORM_TEST_DEVICE", "cuda:6"),
            dispatch_key="AutogradCUDA",
        )
    if name == "cpu":
        return LocalProfile("cpu", "cpu", "cpu", "CPU")
    raise ValueError(
        f"unknown native_layer_norm profile: {name}（内置 thead-ppu / cpu）"
    )
