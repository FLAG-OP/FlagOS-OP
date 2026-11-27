"""Local profile shim for standalone gelu operator execution."""
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
    """PPU 栈识别（本地 shim 用; configs auto_detect 不可靠, 见 thead-ppu.yaml）。

    与 ops/embedding/_profile.py 同款双路探测: triton PPUDriver 优先，
    设备名兜底（PR36 条目 3 口径: 物理卡号）。
    """
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
    """解析 gelu 本地 profile; 缺省按栈身份自动选择。

    - 环境变量 GELU_PROFILE 显式指定
    - 位置参数（example.py / test/*.py CLI 透传）
    - 自动探测顺序: torch_mlu → torch_xmlir → PPU 栈（thead）→ torch_npu → cpu
      （与 embedding/_profile.py 同序, 单平台交付期仅 thead-ppu 有注册实现）
    """
    name = name or os.environ.get("GELU_PROFILE")
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
        # probes/thead_first_probe.py 实测（2026-10-23, 卡6; sdpa/embedding 同源）:
        #   AutogradCUDA 命中 grad/no_grad/aten 直调 6/7 探针（唯一零命中
        #   是 inference_mode, known-issues #19.3 Autograd* 键通病）;
        #   CUDA 键 7/7 全命中。与 p800/T1-sdpa 先例一致选 AutogradCUDA。
        # 设备编号口径（sdpa/PLATFORM.md §3c-2）: 物理卡号 cuda:6。
        return LocalProfile(
            name="thead-ppu",
            vendor="thead",
            torch_device=os.environ.get("GELU_TEST_DEVICE", "cuda:6"),
            dispatch_key="AutogradCUDA",
        )
    if name == "cpu":
        return LocalProfile("cpu", "cpu", "cpu", "CPU")
    raise ValueError(
        f"unknown gelu profile: {name}"
        "（内置 thead-ppu / cpu; 其余平台随交付扩展）"
    )
