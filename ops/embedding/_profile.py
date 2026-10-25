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
    name = name or os.environ.get("EMBEDDING_PROFILE")
    if name is None:
        # 与 sdpa/_profile.py 同序: torch_mlu 优先，再 torch_xmlir，
        # 再 PPU 栈（thead）/ torch_npu，最后 cpu。
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
    if name in ("p800-kunlunxin", "p800", "kunlunxin"):
        return LocalProfile(
            name="p800-kunlunxin",
            vendor="kunlunxin",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "cuda:1"),
            # Dense embedding autograd is intercepted on AutogradCUDA.
            dispatch_key="AutogradCUDA",
        )
    if name in ("ascend910", "ascend"):
        # wt 2026-09-23-fix 第二平台接入: torch_npu 的 A1 拦截点
        # AutogradPrivateUse1（sdpa 开发报告 §3.1 dispatch 证据链）
        return LocalProfile(
            name="ascend910",
            vendor="ascend",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "npu:0"),
            dispatch_key="AutogradPrivateUse1",
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
            # Cambricon A1: 注册到 Autograd key（见上注）
            dispatch_key="AutogradPrivateUse1",
        )
    if name in ("thead-ppu", "thead", "ppu", "thead-ppu-zw810e"):
        # probes/thead_first_probe.py 实测（2026-10-23, 卡6）:
        #   AutogradCUDA 命中 grad/no_grad/aten 直调 6/7 探针（唯一零命中
        #   是 inference_mode, 同 known-issues #19 的 Autograd* 键通病）;
        #   CUDA 键 7/7 全命中。与 p800/T1-sdpa 先例一致选 AutogradCUDA。
        # 设备编号口径（PR36 条目 3）: 物理卡号 cuda:6——与中央 YAML 的
        # torch_device / visible_devices("6,7") 同口径, kernel/op/framework
        # 三层均落在物理卡 6; 取证与换算规则见 PLATFORM.md §5。
        return LocalProfile(
            name="thead-ppu",
            vendor="thead",
            torch_device=os.environ.get("EMBEDDING_TEST_DEVICE", "cuda:6"),
            dispatch_key="AutogradCUDA",
        )
    if name == "cpu":
        return LocalProfile("cpu", "cpu", "cpu", "CPU")
    raise ValueError(
        f"unknown embedding profile: {name}"
        "（内置 p800-kunlunxin / ascend910 / cambricon / thead-ppu / cpu）"
    )
