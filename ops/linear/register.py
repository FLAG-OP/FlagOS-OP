# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# linear 是标准 aten 算子，宿主决定交付路径 = A1（本库集成指南）。
# thead-ppu 注册 key = **基键 CUDA**（中央 configs/devices/thead-ppu.yaml
# 同款分层设计, scripts/smoke_dispatch_key.py 实测基键 grad/no_grad/
# inference_mode 三口径全拦截且反向梯度对齐原生公式）: 委托型算子注册在
# autograd 节点之下——原生 backward 保留（无需 autograd.Function）、
# inference_mode 亦拦截; AutogradCUDA 仅适用于自带反向的算子（如 gelu
# 的 _GeluFn）。PPU 经 CUDA 兼容层直曝 device.type="cuda"。
#
# autograd: 委托原生 aten::linear——原生 autograd 节点随 redispatch 的
# CompositeExplicitAutograd 语义自动建立（linear 的导数公式引用原生
# matmul/mm/add 组合, 不经本注册）, 无需自建 autograd.Function
# （type_as 同款直注册; gelu 需自带反向是因为其注册函数为自研 Triton）。
#
# 调用计数（内存 + 可选文件落地）: 框架级测试跨进程读取, pid 分片避免
# TP 多 worker 并发写竞争（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"linear": 0}
_COUNT_BASE = os.environ.get("A1_LINEAR_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["linear"] = CALL_COUNT.get("linear", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["linear"] = counts.get("linear", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


# ── 平台选择（thead-ppu 首平台; 第二平台出现时按 embedding 范式扩表）──
_BACKENDS = {"thead-ppu": "thead_ppu"}


def _backend_for(dispatch_key: str | None, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（新平台接入见 ops/sdpa/PLATFORM.md §5 范式）")
        return _BACKENDS[platform]
    if dispatch_key in ("AutogradCUDA", "CUDA"):
        return "thead_ppu"
    raise RuntimeError(
        f"未知 dispatch_key={dispatch_key!r}; 已支持: AutogradCUDA/CUDA"
        "（或显式传 platform）")


def _probe_ok(backend: str) -> bool:
    if backend != "thead_ppu":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return any(
            "PPU" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


class _LinearFn(torch.autograd.Function):
    """原生 forward（CUDA redispatch 下探厂商 addmm/mm）+ 组合反向。

    反向用 aten::matmul / sum 组合（不同算子名, 与本注册无递归;
    同进程若亦注册 ops.matmul, 则组合自然路由到 matmul 交付——族行为）。
    linear 的 ATen 导数公式即 matmul/sum 组合（ND 输入先 reshape 2D）。
    1D weight 按矩阵形状 (1,K) 参与，仅在返回前还原其原始形状。
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        try:
            from .kernel.thead_ppu import linear_ppu
        except ImportError:
            from kernel.thead_ppu import linear_ppu
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        return linear_ppu(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        input, weight = ctx.saved_tensors
        with torch.no_grad():
            grad_input, grad_weight, grad_bias = _linear_backward(
                grad_out, input, weight,
                ctx.needs_input_grad[:3], ctx.has_bias)
        return grad_input, grad_weight, grad_bias


def linear_aten(input, weight, bias=None):
    """aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"""
    _bump()
    if torch.is_grad_enabled() and isinstance(input, torch.Tensor) \
            and (input.requires_grad or weight.requires_grad
                 or (bias is not None and bias.requires_grad)):
        return _LinearFn.apply(input, weight, bias)
    try:
        from .kernel.thead_ppu import linear_ppu
    except ImportError:
        from kernel.thead_ppu import linear_ppu
    return linear_ppu(input, weight, bias)


def _linear_backward(grad_out, input, weight, needs_input_grad, has_bias):
    """linear 导数组合（1D 输入/1D weight 均保持原语义）。"""
    matrix_weight = weight.reshape(1, -1) if weight.dim() == 1 else weight
    in2d = input.reshape(-1, input.shape[-1])
    raw_out_shape = in2d.shape[0:1] + matrix_weight.shape[0:1]
    g2d = grad_out.reshape(raw_out_shape)
    grad_input = grad_weight = grad_bias = None
    if needs_input_grad[0]:
        grad_input = (g2d @ matrix_weight).reshape(input.shape)
    if needs_input_grad[1]:
        grad_weight = g2d.t() @ in2d
        if weight.dim() == 1:
            grad_weight = grad_weight.reshape(weight.shape)
    if has_bias and needs_input_grad[2]:
        grad_bias = g2d.sum(0)
    return grad_input, grad_weight, grad_bias


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 linear 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradCUDA（thead 范式, 自带反向——理由与
    递归/autocast 实证同 ops/native_layer_norm/register.py 文件头）。
    counter: 可选 dict, 命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r}, dispatch_key={dispatch_key!r} "
            f"需要可用的 PPU 设备（thead-ppu）")
    fn = linear_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return linear_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("linear", fn, dispatch_key)
    return _LIB


# 统一入口别名（与 A2/B 插件约定对齐，便于框架级注入）
register = register_a1
