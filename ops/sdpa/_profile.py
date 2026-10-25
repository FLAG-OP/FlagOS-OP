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


def _is_ppu_stack() -> bool:
    """PPU 栈识别（本地 shim 用; configs auto_detect 不可靠, 见 thead-ppu.yaml）。"""
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
                name = "thead-ppu" if _is_ppu_stack() else "ascend910"
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
    if name in ("thead-ppu", "thead", "ppu", "thead-ppu-zw810e"):
        # probes/thead_reg_probe.py 实测（2026-10-23, 卡6）:
        #   AutogradCUDA 命中 grad/no_grad/aten 直调 6/7 探针（唯一零命中是
        #   inference_mode, 同 known-issues #19 的 Autograd* 键通病）;
        #   CUDA 键 7/7 全命中但 requires_grad 反传带"未注册 Autograd kernel"
        #   警告。与 p800 先例一致选 AutogradCUDA。
        # 设备编号口径（PR36 条目 3）: 物理卡号 cuda:6——与中央 YAML 的
        # torch_device / visible_devices("6,7") 同口径, kernel/op/framework
        # 三层均落在物理卡 6; 取证与换算规则见 PLATFORM.md §3c-2。
        return LocalProfile(
            name="thead-ppu",
            vendor="thead",
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "cuda:6"),
            dispatch_key="AutogradCUDA",
        )
    raise ValueError(
        "未知设备 profile: "
        f"{name}（内置 ascend910 / p800-kunlunxin / mlu590 / thead-ppu）")


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
        if _is_ppu_stack():
            return load_profile("thead-ppu")
        return load_profile("ascend910")
