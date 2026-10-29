"""Multi-platform facade for ``aten::embedding``.

Mirrors ``ops/sdpa/kernel/triton_level.py``: callers keep a single import and
the platform backend is chosen at call time from ``tensor.device.type``.

- ``mlu``  -> [kernel/cambricon.py](cambricon.py) (Cambricon MLU590, torch_mlu)
- ``cuda`` -> [kernel/p800_kunlunxin.py](p800_kunlunxin.py) (XMLIR presents
  the Kunlunxin XPU as a CUDA device) 或
  [kernel/thead_ppu.py](thead_ppu.py) (T-head PPU ZW810E, CUDA 兼容层)——
  栈身份按 torch_xmlir / PPU 设备名探测, 进程级缓存（T1 sdpa
  backends/__init__.py 同款, 见 MERGE.md thead 节）
- ``npu``  -> [kernel/ascend910.py](ascend910.py) (Ascend 910, torch_npu;
  wt 2026-10-07-fix 第三平台并入 facade——#11 的 backend 接入 #9 的
  选择器, 保持"单一 import、按设备类型路由"范式)

Tests, benchmark scripts, and the A1 registration in ``register.py`` import
from this module so a single code path serves both platforms.
"""
from __future__ import annotations

import importlib.util

import torch

try:  # Package-style import: ops.embedding.kernel.platform
    from . import cambricon as _cambricon
    from . import p800_kunlunxin as _p800
    from . import ascend910 as _ascend
    from . import thead_ppu as _thead
except ImportError:  # Standalone import with OP_DIR on sys.path
    from kernel import cambricon as _cambricon
    from kernel import p800_kunlunxin as _p800
    from kernel import ascend910 as _ascend
    from kernel import thead_ppu as _thead


PLATFORM = "multi(ascend910,cambricon,p800-kunlunxin,thead-ppu)"
SUPPORTED_DEVICE_TYPES = ("mlu", "cuda", "npu")

_BACKENDS = {"mlu": _cambricon, "npu": _ascend}
# "cuda" 有两候选（昆仑芯 XMLIR 伪装 CUDA / T-head PPU CUDA 兼容层）,
# 栈身份进程内不变 → 进程级缓存; 不加缓存则每次 embedding() 调用都要
# find_spec/import, 与 T1 sdpa 的 80-100µs/调用 路由开销教训同因。
_CUDA_BACKEND = None


def _cuda_impl():
    global _CUDA_BACKEND
    if _CUDA_BACKEND is not None:
        return _CUDA_BACKEND
    # XMLIR 先判（p800 backend 内部同样有 torch_xmlir 守卫）。
    if importlib.util.find_spec("torch_xmlir") is not None:
        try:
            import torch_xmlir  # noqa: F401
            _CUDA_BACKEND = _p800
            return _CUDA_BACKEND
        except Exception:
            pass
    if _thead._has_ppu_stack():
        _CUDA_BACKEND = _thead
        return _CUDA_BACKEND
    raise RuntimeError(
        "embedding 的 cuda 设备需要 torch_xmlir（P800/Kunlunxin）或 "
        "PPU 栈（T-head ZW810E），当前环境两者皆无（见 MERGE.md）"
    )


# A1 拦截 key 与设备类型对应（实测结论见 reports/cambricon.md §4）:
# 两个平台都必须注册到各自的 Autograd key，autograd.Function 才能同时
# 覆盖 grad/no_grad 两种调用（p800=AutogradCUDA，MLU=AutogradPrivateUse1）。
# wt 2026-10-07-fix ascend910: torch_npu 的 A1 拦截点同为
# AutogradPrivateUse1（sdpa 开发报告 §3.1 dispatch 证据链; 与 mlu
# 同 key, 单机双 NPU+MLU 场景须 profile 显式指定平台）
# thead-ppu: probes/thead_first_probe.py 实测 AutogradCUDA 6/7 命中
# （inference_mode 零命中 = known-issues #19 Autograd 键通病, 与 p800/T1
# sdpa 同结论）, 与 p800 同 key。
# # wt <wangt635@ustc.edu.cn>
DISPATCH_KEYS = {"cuda": "AutogradCUDA", "mlu": "AutogradPrivateUse1",
                  "npu": "AutogradPrivateUse1"}


def get_backend(device_type: str):
    """Return the platform backend module for a torch device type."""
    if device_type == "cuda":
        return _cuda_impl()
    try:
        return _BACKENDS[device_type]
    except KeyError:
        raise RuntimeError(
            f"embedding 不支持 device_type={device_type!r}，"
            f"已知平台设备类型: {sorted(_BACKENDS) + ['cuda']}"
        ) from None


def dispatch_key_for(device: str | torch.device) -> str:
    """设备字符串 -> A1 拦截 key（bench 脚本没有 profile 时用）。"""
    device_type = str(device).split(":")[0]
    if device_type not in DISPATCH_KEYS:
        raise RuntimeError(
            f"没有 {device_type!r} 对应的 A1 dispatch key，"
            f"已知: {DISPATCH_KEYS}"
        )
    return DISPATCH_KEYS[device_type]


def synchronize(device: str | torch.device) -> None:
    """平台同步（计时口径: kernel 完成后再读回/再取时间）。"""
    device_type = str(device).split(":")[0]
    if device_type == "mlu":
        torch.mlu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "npu":
        torch.npu.synchronize()
    else:
        raise RuntimeError(
            f"没有 {device_type!r} 对应的 synchronize 实现"
        )


def embedding(
    weight: torch.Tensor,
    indices: torch.Tensor,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """``aten::embedding`` routed to the backend of ``weight.device``."""
    return get_backend(weight.device.type).embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )


def embedding_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int | None = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """Dense embedding backward routed by ``grad_output.device``."""
    return get_backend(grad_output.device.type).embedding_backward(
        grad_output, indices, num_weights, padding_idx,
        scale_grad_by_freq, sparse,
    )
