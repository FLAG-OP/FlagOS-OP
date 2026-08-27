# audit 厂商算子: 计数 + 委托。委托目标来自设备 profile。
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.base import Backend


_COUNT_BASE = os.environ.get("AUDIT_VENDOR_COUNT_FILE",
                              "/tmp/audit_vendor_counts")
# pid 后缀: 消除 TP 多 worker 并发写同一文件的竞争；读取方用
# read_counts() 汇总全部 pid 分片
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}"


def _bump(op: str) -> None:
    counts = {}
    try:
        with open(COUNT_FILE) as f:
            counts = json.load(f)
    except (OSError, ValueError):
        pass
    counts[op] = counts.get(op, 0) + 1
    with open(COUNT_FILE, "w") as f:
        json.dump(counts, f)


def read_counts() -> dict:
    """汇总全部 pid 分片的计数（消除并发写竞争后的读取端）。"""
    import glob as _glob
    total: dict = {}
    for p in _glob.glob(f"{_COUNT_BASE}.*"):
        try:
            with open(p) as f:
                for k, v in json.load(f).items():
                    total[k] = total.get(k, 0) + v
        except (OSError, ValueError):
            continue
    return total


def _vendor_delegate(pkg: str, func: str):
    import importlib
    mod = importlib.import_module(pkg)
    return getattr(mod, func)


class AuditVendorBackend(Backend):
    """vendor 名为 'audit' 的演示 backend。

    silu_and_mul 双模式委托（AUDIT_DELEGATE 环境变量控制）:
      vendor    —— 调用设备 profile 指定的厂商 kernel（纯拦截零数值变化，
                   框架级测试用，输出必须与基线一致）
      reference —— 委托 vllm_fl reference 实现（语义明确，算子层精度测试用）
    profile 无 vendor_delegate 时 vendor 模式自动退化为 reference。
    """

    @property
    def name(self) -> str:
        return "audit"

    @property
    def vendor(self) -> Optional[str]:
        return "audit"

    def is_available(self) -> bool:
        return True

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        _bump("silu_and_mul")
        delegate = os.environ.get("AUDIT_DELEGATE", "vendor")

        if delegate == "vendor":
            spec = os.environ.get("AUDIT_VENDOR_DELEGATE", "")  # "pkg.func"
            if spec and "." in spec:
                pkg, func = spec.rsplit(".", 1)
                try:
                    vendor_fn = _vendor_delegate(pkg, func)
                    d = x.shape[-1] // 2
                    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype,
                                      device=x.device)
                    vendor_fn(x, out)
                    return out
                except ImportError:
                    pass  # 厂商库不存在 → 退化为 reference
        # reference 委托
        from vllm_fl.dispatch import get_default_manager
        impl = get_default_manager().registry.get_implementation(
            "silu_and_mul", "reference.torch")
        return impl.fn(obj, x)
