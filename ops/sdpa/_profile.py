# 本地设备 profile 垫片——与 FlagOS-OP common/device.py 的 DeviceProfile
# 同接口。standalone 运行时内置 ascend910 / p800-kunlunxin，并可用
# SDPA_PROFILE 显式指定。
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
    if name is None:
        name = os.environ.get("SDPA_PROFILE")
    if name is None:
        try:
            import torch_mlu  # noqa: F401
            name = "mlu590"
        except ImportError:
            try:
                import torch_xmlir  # noqa: F401
                name = "p800-kunlunxin"
            except ImportError:
                name = "ascend910"
    if name in (None, "ascend910", "ascend910b"):
        return LocalProfile(
            name="ascend910",
            vendor="ascend",
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "npu:0"),
            dispatch_key="AutogradPrivateUse1",  # torch_npu 实测拦截点
        )
    if name in ("p800-kunlunxin", "p800", "kunlunxin"):
        return LocalProfile(
            name="p800-kunlunxin",
            vendor="kunlunxin",
            # CUDA_VISIBLE_DEVICES=1,2 中内训实机为 cuda:1；cuda:0 曾触发
            # 厂商 SDPA 挂起，保持与设备 profile/环境锁一致。
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "cuda:1"),
            dispatch_key="AutogradCUDA",
        )
    if name in ("mlu590", "mlu", "cambricon", "cambricon-mlu590"):
        return LocalProfile(
            name="mlu590",
            vendor="cambricon",
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "mlu:0"),
            dispatch_key="AutogradPrivateUse1",  # torch_mlu 实测拦截点
        )
    raise ValueError(
        "未知设备 profile: "
        f"{name}（内置 ascend910 / p800-kunlunxin / mlu590）")


def detect_profile() -> LocalProfile:
    try:
        import torch_mlu  # noqa: F401
        return load_profile("mlu590")
    except ImportError:
        pass
    try:
        import torch_xmlir  # noqa: F401
        return load_profile("p800-kunlunxin")
    except ImportError:
        return load_profile("ascend910")
