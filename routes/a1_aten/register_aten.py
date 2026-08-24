# aten dispatcher 注册模板（设备无关: dispatch key 由 profile 提供）
from __future__ import annotations

import json
import os
import pathlib
from typing import Optional

import torch


# 调用计数（内存 + 可选文件落地，框架级测试跨进程读取）
CALL_COUNT = {"gelu": 0, "silu": 0}
COUNT_FILE = os.environ.get("A1_ATEN_COUNT_FILE")  # 设置后每次调用写入文件


def _bump(op: str) -> None:
    CALL_COUNT[op] = CALL_COUNT.get(op, 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts[op] = counts.get(op, 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


def _load_kernels():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "a1_kernels", pathlib.Path(__file__).parent / "kernels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = None


def gelu_with_fallback(self, *, approximate="none"):
    """aten::gelu: (Tensor self, *, str approximate='none')"""
    global _K
    if _K is None:
        _K = _load_kernels()
    if self.is_cuda and approximate == "tanh":
        _bump("gelu")
        return _K.gelu_tanh_triton(self)
    return torch.nn.functional.gelu(self, approximate=approximate)


def silu_identity_counter(self):
    """aten::silu 数值恒等覆盖（x*sigmoid(x)，同 dtype 同数学）。

    框架级测试用: fp32 计算后 cast（与 ATen CUDA kernel 的
    acc_type<float> 行为一致）→ 位级等价 → 端到端输出必须相同，
    """
    _bump("silu")
    xf = self.float()
    return (xf * torch.sigmoid(xf)).to(self.dtype)


def register_gelu_aten(dispatch_key: str) -> torch.library.Library:
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("gelu", gelu_with_fallback, dispatch_key)
    return lib


def register_silu_aten(dispatch_key: str) -> torch.library.Library:
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("silu", silu_identity_counter, dispatch_key)
    return lib


if __name__ == "__main__":
    _lib = register_gelu_aten("CUDA")
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda:0")
    torch.nn.functional.gelu(x, approximate="tanh")
    print("gelu intercepted:", CALL_COUNT["gelu"])
