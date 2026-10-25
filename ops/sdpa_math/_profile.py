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


def _is_ppu_stack() -> bool:
    """PPU 栈识别（configs auto_detect 不可靠——T1 thead-ppu.yaml 同款）。"""
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
        name = os.environ.get("SDPA_MATH_PROFILE")
    if name is None:
        try:
            import torch
            if hasattr(torch, "npu") and torch.npu.is_available():
                name = "ascend910"
            elif _is_ppu_stack():
                name = "thead-ppu"
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
    if name in ("thead-ppu", "thead", "ppu", "thead-ppu-zw810e"):
        # probes/thead_first_probe.py 实测（2026-10-24, 卡6）:
        #   AutogradCUDA 6/7 命中（唯一零命中 inference_mode =
        #   known-issues #19 Autograd 键通病, 与 T1 sdpa / T2 embedding 同）;
        #   CUDA 键 7/7 全命中但反传带"未注册 Autograd kernel"警告。
        #   与 p800/T1 先例一致选 AutogradCUDA。
        # 设备编号口径（PR36 条目 3）: 物理卡号 cuda:6——与中央 YAML 的
        # torch_device / visible_devices("6,7") 同口径, kernel/op/framework
        # 三层均落在物理卡 6; 取证与换算规则见 PLATFORM.md §3a 表。
        return LocalProfile(
            name="thead-ppu",
            vendor="thead",
            torch_device=os.environ.get("SDPA_MATH_TEST_DEVICE", "cuda:6"),
            dispatch_key="AutogradCUDA",
            default_impl="thead",  # flash/efficient 分流 + exact-P
        )
    raise ValueError(
        f"未知设备 profile: {name}"
        f"（内置 cpu / ascend910 / p800-kunlunxin / thead-ppu）")


def detect_profile() -> LocalProfile:
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            return load_profile("ascend910")
        if _is_ppu_stack():
            return load_profile("thead-ppu")
        if torch.cuda.is_available():
            try:
                import torch_xmlir  # noqa: F401
                return load_profile("p800-kunlunxin")
            except ImportError:
                pass
    except Exception:
        pass
    return load_profile("cpu")
