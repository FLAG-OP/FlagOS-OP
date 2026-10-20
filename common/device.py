# 设备 Profile 加载与自动探测
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ascend(torch_npu) 排序陷阱: 镜像默认 TASK_QUEUE_ENABLE=1，torch_npu 把
# 算子排入 task queue 异步执行，而 triton-ascend 直接在 raw stream 上启动
# kernel（见 triton/backends/ascend/backend_register.py::get_current_stream）。
# 两者之间**没有顺序保证**——生产者算子（randn/.to/mul）尚未完成，自研
# Triton kernel 就读到未初始化输入，表现为非确定性错值/NaN（本库实测：
# 同一输入两次调用结果不同、有时对有时错）。A1 自研 kernel 依赖输入就绪，
# 且必须在 torch_npu 初始化之前生效，故在此强制关闭 task queue。
# 仅在安装了 torch_npu 的机器上生效（find_spec 不导入 torch），不影响
# MLU/CUDA/CPU 机器。Dashener2 2026-10-15-fix 首现于布局/分配算子 Ascend 移植。
# # Dashener2 <dashen272@outlook.com>
if importlib.util.find_spec("torch_npu") is not None:
    os.environ["TASK_QUEUE_ENABLE"] = "0"

import torch  # noqa: F401  # 必须早于 xpu_compat(→triton) 导入：MLU 栈
# 要求先 import torch，否则 torch.library 的 triton 命名空间重复注册报错
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
    auto_detect_enabled: bool = True
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
        auto_detect_enabled=bool(
            raw.get("auto_detect", {}).get("enabled", True)
        ),
        golden_engine=raw.get("golden_engine", "vllm"),
        raw=raw,
    )


def detect_profile() -> DeviceProfile:
    """按 flag_gems vendor 探测 + auto_detect.module 探测当前设备。

    探测顺序:
      1. 各 profile 的 auto_detect.module 可 import → 选中
      2. flag_gems DeviceDetector().vendor_name 匹配 profile.vendor → 选中
      3. 失败则报错并提示 --device 指定

    仅 auto_detect.enabled 不为 false 的 profile 进入两条探测路径的候选；
    禁用项仍可用 --device / load_profile() 显式选择。自动探测只选择默认
    硬件配置，不校验软件栈版本——同硬件多软件栈（如 P800 4.2/5.0）必须
    显式指定 profile，软件版本用 scripts/check_env.py 核对。
    """
    profiles = [
        p for p in (load_profile(n) for n in list_profiles())
        if p.auto_detect_enabled
    ]

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
