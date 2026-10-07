# 本地设备 profile 垫片——与 FlagOS-OP common/device.py 的 DeviceProfile
# 同接口。standalone 运行时内置 cpu / ascend910 / p800-kunlunxin，并可用
# SDPA_MATH_PROFILE
# 显式指定。dispatch_key 是 A1 注册点（见 register.py 的实测说明）。
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LocalProfile:
    name: str
    vendor: str
    torch_device: str
    dispatch_key: str
    default_impl: str

    def summary(self) -> str:
        return (f"{self.name} [vendor={self.vendor} "
                f"dev={self.torch_device} key={self.dispatch_key} "
                f"impl={self.default_impl}]")


def load_profile(name: str | None = None) -> LocalProfile:
    if name is None:
        name = os.environ.get("SDPA_MATH_PROFILE")
    if name is None:
        try:
            import torch
            if hasattr(torch, "npu") and torch.npu.is_available():
                name = "ascend910"
            elif torch.cuda.is_available():
                try:
                    import torch_xmlir  # noqa: F401
                    name = "p800-kunlunxin"
                except ImportError:
                    name = "cpu"
            else:
                name = "cpu"
        except Exception:
            name = "cpu"
    if name in ("ascend910", "ascend910b", "npu"):
        return LocalProfile(
            name="ascend910",
            vendor="ascend",
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "npu:0"),
            dispatch_key="AutogradPrivateUse1",  # torch_npu 实测拦截点
            default_impl="triton",               # 自研 Triton 设备码
        )
    if name == "cpu":
        return LocalProfile(
            name="cpu",
            vendor="cpu",
            torch_device="cpu",
            dispatch_key="CPU",  # configs/devices/cpu.yaml 一致
            default_impl="torch",  # 自研 ATen 组合（无设备码时的默认）
        )
    if name in ("p800-kunlunxin", "p800", "kunlunxin"):
        return LocalProfile(
            name="p800-kunlunxin",
            vendor="kunlunxin",
            # cuda:0 on this shared image has shown vendor runtime hangs;
            # keep the stable card used by the repository P800 profile.
            torch_device=os.environ.get("SDPA_MATH_TEST_DEVICE", "cuda:1"),
            dispatch_key="AutogradCUDA",   # paired with CUDA by register_a1
            default_impl="p800",            # vendor O/LSE + exact-P composition
        )
    raise ValueError(
        f"未知设备 profile: {name}（内置 cpu / ascend910 / p800-kunlunxin）")


def detect_profile() -> LocalProfile:
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            return load_profile("ascend910")
        if torch.cuda.is_available():
            try:
                import torch_xmlir  # noqa: F401
                return load_profile("p800-kunlunxin")
            except ImportError:
                pass
    except Exception:
        pass
    return load_profile("cpu")
