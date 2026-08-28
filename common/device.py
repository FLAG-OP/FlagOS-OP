# 设备 Profile 加载与自动探测
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# 尽早生效: 让 XPU Triton 编译失败暴露真实原因（known-issues #14）
from common.xpu_compat import ensure_xpu_compiler_debuggable  # noqa: E402

ensure_xpu_compiler_debuggable()

DEVICES_DIR = Path(__file__).resolve().parents[1] / "configs" / "devices"


@dataclass
class DeviceProfile:
    name: str
    vendor: str
    torch_device: str
    visible_devices: str
    dispatch_key: str
    framework: dict
    vendor_delegate: Optional[dict]
    reference_fallback: str
    auto_detect_module: Optional[str] = None
    golden_engine: str = "vllm"       # vllm | transformers
    raw: dict = field(default_factory=dict)

    # ---------- 便捷访问 ----------
    @property
    def framework_quirks(self) -> dict:
        return self.framework.get("quirks", {})

    def engine_args(self) -> dict:
        """vLLM LLM(...) 引擎参数（去掉 quirks，model_path→model）。"""
        args = {k: v for k, v in self.framework.items() if k != "quirks"}
        if "model_path" in args:
            args["model"] = args.pop("model_path")
        return args

    def delegate_import(self) -> tuple[Optional[str], Optional[str]]:
        """vendor_delegate "pkg.func" -> ("pkg", "func")；null -> (None, None)。"""
        if not self.vendor_delegate:
            return None, None
        s = str(self.vendor_delegate)
        if "." not in s:
            return s, None
        pkg, func = s.rsplit(".", 1)
        return pkg, func

    def summary(self) -> str:
        return (f"{self.name} [vendor={self.vendor} dev={self.torch_device} "
                f"visible={self.visible_devices} key={self.dispatch_key}]")

    @property
    def golden_engine_hint(self) -> str:
        return str(self.raw.get("golden_engine", "vllm"))


def list_profiles() -> list[str]:
    """列出可用 profile 名（跳过 _template）。"""
    return sorted(
        p.stem for p in DEVICES_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    )


def load_profile(name: str) -> DeviceProfile:
    path = DEVICES_DIR / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(list_profiles()) or "(无)"
        raise FileNotFoundError(
            f"设备 profile '{name}' 不存在 ({path})。可用: {available}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    dev = raw.get("device", {})
    fw = raw.get("framework", {})
    # 模型路径可被环境变量覆盖（换机器/换模型不改 profile）
    if os.environ.get("FLAGOS_MODEL_PATH"):
        fw["model_path"] = os.environ["FLAGOS_MODEL_PATH"]
    return DeviceProfile(
        name=raw["name"],
        vendor=raw["vendor"],
        torch_device=dev.get("torch_device", "cuda:0"),
        visible_devices=str(dev.get("visible_devices", "0")),
        dispatch_key=dev.get("dispatch_key", "CUDA"),
        framework=fw,
        vendor_delegate=raw.get("vendor_delegate"),
        reference_fallback=raw.get("reference_fallback", "reference.torch"),
        auto_detect_module=raw.get("auto_detect", {}).get("module"),
        golden_engine=raw.get("golden_engine", "vllm"),
        raw=raw,
    )


def detect_profile() -> DeviceProfile:
    """按 flag_gems vendor 探测 + auto_detect.module 探测当前设备。

    探测顺序:
      1. 各 profile 的 auto_detect.module 可 import → 选中
      2. flag_gems DeviceDetector().vendor_name 匹配 profile.vendor → 选中
      3. 失败则报错并提示 --device 指定
    """
    profiles = [load_profile(n) for n in list_profiles()]

    # 路径1: 模块可导入
    import importlib
    for p in profiles:
        if p.auto_detect_module and p.auto_detect_module != "torch_cuda_vendor_probe":
            try:
                importlib.import_module(p.auto_detect_module)
                return p
            except ImportError:
                continue

    # 路径2: flag_gems vendor 匹配（nvidia 等标准 CUDA 走这里）
    try:
        from flag_gems.runtime.backend.device import DeviceDetector
        vendor = DeviceDetector().vendor_name
        for p in profiles:
            if p.vendor == vendor:
                return p
    except Exception:
        pass

    available = ", ".join(list_profiles()) or "(无)"
    raise RuntimeError(
        f"无法自动探测设备 profile。请用 --device 显式指定。可用: {available}"
    )
