"""T-head PPU (ZW810E) SDPA backend.

Engineering choice
------------------
按 PLATFORM.md §3 p800 厂商委托先例: 厂商 aten overload 直调，绕开 Python
``F.sdpa`` monkey patch 以免 A1 注册后递归。fp16/bf16 按探针实测择优分流
（probes/thead_flash_probe.py）:

- **maskless（attn_mask is None, dropout_p=0）→ flash 直调**
  ``aten::_scaled_dot_product_flash_attention``。TorchDispatchMode 实证
  PPU 原生 ``F.sdpa`` 内部恰好只调这一个 overload——直调即与原生同
  kernel；最终 perf_run 台账 0.114 ms vs 原生 0.104 ms（1.10×,
  37.7 TFLOPS）。flash 输出连续（efficient 输出 D-major 非连续，
  ``reshape(-1)`` 会整份拷贝）。
- **带 mask → efficient 委托**（flash schema 无 attn_mask 参数，
  实测 ``Unknown keyword argument 'attn_mask'``）。masked 形状
  bench 实测 1.08× 原生。

首版全量走 efficient 时出现 4.07× 性能倒挂（基线台账 native 0.105 ms/
41.07 TFLOPS vs efficient 0.426 ms/10.08 TFLOPS），根因两个、均已收口:
(1) masked 之外全走 efficient 且其输出非连续; (2) ``backends.get_impl``
每次调用 ``find_spec``/``importlib`` 解析三候选（cProfile 实测
~80-100 µs/调用）——已加进程级 ``_IMPL_CACHE``。

- cudnn overload 在 PPU 上不可用（实测 ``CUDNN_BACKEND_TENSOR_DESCRIPTOR:
  ... cudnn_status: ACDNN_STATUS_BAD_PARAM``），生产路径不选。
- fp32 委托精度本身很好（非 extreme 案例 3e-7 ～ 5e-6），但
  scale=8 饱和（黄金 extreme 口径）实测 abs 2.79e-3 超判卷线，故按
  p800 先例 **fp32 走 ATen 组合**（``sdpa_reference`` 设备端 fp32 组合）。
  PPU 无 XMLIR bmm JIT 缺陷——实测 S∈(320,640]（p800 的 768 pad 区间）
  设备组合 vs CPU 参考 ≤5.7e-7，无需 pad 绕过。

与 p800 先例的实测差异（probes/thead_eff_probe.py、thead_bias_layout.py）:

1. efficient schema **无 ``enable_gqa``**（8 参数实测）——GQA 用
   ``repeat_interleave`` 扩 KV（mlu590 overrideable 同款做法）。
2. ``attn_bias`` 存在 16B stride 对齐约束（两条实测规则，对连续 bias）:
   ``(stride(1)*elem) % 16 == 0`` 报 ``attn_bias is not correctly aligned
   (strideH)``；``(stride(2)*elem) % 16 == 0`` 报 ``... aligned``（无后缀）。
   不满足时把 bias 复制进按列填充到 16B 对齐步长的缓冲区，再
   ``as_strided`` 回原逻辑形状（K-stride 保持 1）——实测 9/9 语义等价。

备选（未接入生产路径）: triton 3.6.0 在 PPU 可编译执行（target
cuda arch=80, PPUDriver），但 ``tl.dot`` 实测精度掉到 fp16 量级
（64³ 点积 rel≈7e-4；``input_precision="ieee"`` K=128 abs 1.03e-4）
仍超黄金 fp32 1e-5 绝对容差——故 triton ieee 仅记为编译可行的备选，
不接生产（见 probes/thead_triton_dot_smoke.py）。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

try:  # Package-style import from repository root / multi-operator process.
    from ...reference import sdpa_reference
except ImportError:  # Standalone ops/sdpa execution inserts OP_DIR in sys.path.
    from reference import sdpa_reference


PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)   # PPU torch 经 CUDA 兼容层直曝 cuda:N


def _has_ppu_stack(device: torch.device) -> bool:
    """区分 PPU 的 CUDA 兼容层与 NVIDIA CUDA / 昆仑芯 XMLIR。

    XMLIR 另有 torch_xmlir 守卫（见 p800 backend）；此处以厂商设备名为
    准（实测 16 卡均为 PPU-ZW810E）。
    """
    if device.type != "cuda":
        return False
    try:
        return "PPU" in torch.cuda.get_device_name(device)
    except Exception:
        return False


def _aligned_bias(bias: torch.Tensor) -> torch.Tensor:
    """修复 PPU efficient kernel 的 attn_bias 16B stride 对齐约束。

    行距对齐后头距（= Sq × 行距）自动对齐；逻辑形状与 K-stride=1 不变，
    kernel 读取范围仍为原 (Sq, Skv)。
    """
    elem = bias.element_size()
    if (bias.stride(2) * elem) % 16 == 0 and (bias.stride(1) * elem) % 16 == 0:
        return bias
    B, H, Sq, Skv = bias.shape
    step = 16 // elem
    skv_pad = (Skv + step - 1) // step * step
    buf = bias.new_zeros(B, H, Sq, skv_pad)
    buf[..., :Skv] = bias
    return buf.as_strided((B, H, Sq, Skv),
                          (H * Sq * skv_pad, Sq * skv_pad, skv_pad, 1))


def _expand_gqa(query, key, value, enable_gqa):
    """厂商 flash/efficient schema 均无 enable_gqa——GQA 扩 KV（同 mlu590）。"""
    if enable_gqa:
        rep = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    return query, key, value


def _flash_sdpa(query, key, value, is_causal, scale, enable_gqa):
    """maskless 快路径: flash overload 直调（schema 无 mask, 仅 causal/full）。"""
    query, key, value = _expand_gqa(query, key, value, enable_gqa)
    return torch.ops.aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        0.0,           # dropout_p（入口已强制 0）
        is_causal,
        False,         # return_debug_mask
        scale=scale,
    )[0]


def _efficient_sdpa(query, key, value, attn_mask, dropout_p, is_causal,
                    scale, enable_gqa):
    """厂商委托: 直调 private aten efficient 入口。"""
    bias = attn_mask
    if bias is not None and bias.dtype == torch.bool:
        # efficient kernel 只收加性 float bias, 不收 bool（p800 先例）
        bias = torch.zeros_like(bias, dtype=query.dtype)
        bias.masked_fill_(~attn_mask, float("-inf"))
    elif bias is not None and bias.dtype != query.dtype:
        bias = bias.to(query.dtype)
    if bias is not None:
        bias = _aligned_bias(bias)

    query, key, value = _expand_gqa(query, key, value, enable_gqa)

    # 与 p800 同款: 仅当直调路径的 autograd 真的能消费 log-sumexp 时才
    # 计算；A1 自定义 Function 的 forward 处于 no-grad，反向由数学实现承担。
    compute_logsumexp = torch.is_grad_enabled() and any((
        query.requires_grad,
        key.requires_grad,
        value.requires_grad,
    ))
    out = torch.ops.aten._scaled_dot_product_efficient_attention(
        query,
        key,
        value,
        attn_bias=bias,
        compute_log_sumexp=compute_logsumexp,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )[0]
    if attn_mask is not None:
        # 全遮蔽行: CPU/aten 语义为 NaN; 厂商 kernel 分化行为统一恢复
        # （p800 #17.3 / mlu590 #18.4 同款处理）。
        if attn_mask.dtype == torch.bool:
            effective_false = ~attn_mask.bool()
        else:
            effective_false = torch.isneginf(attn_mask.float())
        if is_causal:
            sq, skv = query.shape[-2], key.shape[-2]
            causal = torch.ones(
                (sq, skv), dtype=torch.bool, device=query.device
            ).tril()
            effective_false = effective_false | ~causal
        all_masked = effective_false.all(dim=-1, keepdim=True)
        out = out.masked_fill(all_masked, float("nan"))
    return out


def _sdpa_float32_exact(query, key, value, attn_mask, dropout_p, is_causal,
                        scale, enable_gqa):
    """fp32 ATen 组合（p800 §3 先例; PPU 无需 bmm pad 绕过）。"""
    return sdpa_reference(
        query, key, value, attn_mask, dropout_p, is_causal, scale,
        enable_gqa,
    )


def _mask_grad_path(query, key, value, attn_mask, dropout_p, is_causal,
                    scale, enable_gqa):
    """可微 additive mask: 复用 A1 数学 backward（无递归——被本函数调用时
    autograd 已关闭, 不会再进此分支）。"""
    try:  # Package-style import: ops.sdpa.kernel.backends.thead_ppu
        from ...register import _SDPA_A1_Function
    except ImportError:  # Standalone import with OP_DIR on sys.path
        from register import _SDPA_A1_Function

    saved_impl = getattr(_SDPA_A1_Function, "_impl", "triton")
    _SDPA_A1_Function._impl = "triton"
    try:
        return _SDPA_A1_Function.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )
    finally:
        _SDPA_A1_Function._impl = saved_impl


def sdpa_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **_: Any,
) -> torch.Tensor:
    """aten-compatible SDPA entry for T-head PPU.

    Legacy tuning keyword arguments (block_m/block_n/num_warps/...) are
    accepted and ignored because scheduling is delegated to the vendor kernel.
    """
    if query.device.type not in SUPPORTED_DEVICE_TYPES or \
            not _has_ppu_stack(query.device):
        raise RuntimeError(
            f"sdpa_triton 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={query.device!r} 或当前栈非 PPU（设备名需含 "
            f"PPU）。请按平台选择对应实现（见 PLATFORM.md）。"
        )
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("仅支持 4D (B, H, S, D)")
    if dropout_p != 0.0:
        raise ValueError("不支持 dropout（确定性验证路径）")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"不支持的 dtype: {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query/key/value dtype 不一致")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("head_dim 不一致")
    if is_causal and query.shape[-2] != key.shape[-2]:
        raise ValueError("causal 语义要求 Sq == Skv（与 aten/reference 一致）")

    hq, hkv = query.shape[1], key.shape[1]
    if attn_mask is not None:
        if attn_mask.dim() != 4:
            raise ValueError("仅支持 4D attn_mask")
        expected_mask_shape = (
            query.shape[0], hq, query.shape[-2], key.shape[-2]
        )
        if attn_mask.shape != expected_mask_shape:
            raise ValueError(
                f"attn_mask shape 应为 {expected_mask_shape}，"
                f"收到 {tuple(attn_mask.shape)}"
            )
        if (attn_mask.dtype != torch.bool
                and not attn_mask.dtype.is_floating_point):
            raise ValueError("attn_mask 仅支持 bool 或浮点加性 bias")
    if hq != hkv:
        if not enable_gqa:
            raise ValueError("Hq != Hkv 需要 enable_gqa=True")
        if hq % hkv:
            raise ValueError("GQA 要求 Hq 整除 Hkv")
    if query.shape[-2] == 0 or key.shape[-2] == 0:
        return torch.empty_like(query)

    if (attn_mask is not None and torch.is_grad_enabled()
            and attn_mask.requires_grad):
        return _mask_grad_path(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )

    if query.dtype == torch.float32:
        return _sdpa_float32_exact(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa,
        )
    if attn_mask is None and dropout_p == 0.0:
        # maskless: flash 与原生 F.sdpa 同速（efficient 有 ~2.1× 倒挂,
        # 见模块 docstring / reports/performance.md）
        return _flash_sdpa(
            query, key, value, is_causal, scale, enable_gqa,
        )
    return _efficient_sdpa(
        query, key, value, attn_mask, dropout_p, is_causal, scale,
        enable_gqa,
    )
