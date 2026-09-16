# 本地设备 profile 垫片——与 FlagOS-OP common/device.py 的 DeviceProfile
# 同接口。standalone 运行（不在 FlagOS-OP 仓库内）时使用内置 Ascend 910
# profile；设置 FLAGOS_OP_ROOT 时优先用仓库的 profile 体系。
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
    root = os.environ.get("FLAGOS_OP_ROOT")
    if root and not name:
        name = "ascend910"          # 仓库内已建同名 profile 时走仓库
    if name in (None, "ascend910", "ascend910b"):
        return LocalProfile(
            name="ascend910",
            vendor="ascend",
            torch_device=os.environ.get("SDPA_TEST_DEVICE", "npu:0"),
            dispatch_key="PrivateUse1",   # torch_npu 的 aten 后端 key
        )
    raise ValueError(f"未知设备 profile: {name}（本目录仅内置 ascend910）")


def detect_profile() -> LocalProfile:
    return load_profile()
