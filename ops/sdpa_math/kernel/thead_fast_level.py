# thead-ppu (T-head PPU ZW810E) fast path for exact sdpa_math contract.
#
# Fork decision (ops/sdpa/MERGE.md §0 → structural): sdpa_math 必须物化第二
# 输出 P，不能整体替换为 flash/efficient；与 p800 先例同构——vendor O/LSE
# + exact-P 补偿。与 p800 的差异（首测 probes/thead_first_probe.py）:
#
#   1. maskless 且（非 causal 或 Sq==Skv）走 flash 直调（T1 sdpa 同款——
#      原生 F.sdpa 内部即 flash；fp16/bf16）。**PPU flash 对非方阵 causal
#      （Sq≠Skv）不生效**（首测 flash O err≈2.0 vs efficient err≈0）——
#      非方阵 causal 与带 mask 场景走 efficient（flash schema 无 attn_bias）。
#   2. PPU efficient 的 LSE 按 32 倍数 pad（Sq=16→32, 100→128, 197→224），
#      materialize P 前必须切片到 Sq；flash LSE 本身即 (B,H,Sq)，无 pad。
#      （p800 XMLIR 栈 LSE 无 pad——该差异为 thead 专属绑定。）
#   3. flash 仅支持 fp16/bf16（fp32 直调报
#      "FlashAttention only support fp16 and bf16 data type"）；fp32 走
#      torch_level ATen 组合（extreme scale=8 委托 abs≈2e-5 超 1e-5，
#      T1/p800 先例同判）。
#   4. GQA: flash/efficient schema 均无 enable_gqa → repeat_interleave 扩 KV。
#   5. bool mask → additive -inf bias；16B stride 不齐时 _aligned_bias 列 pad
#      + as_strided（T1 sdpa 同款约束）。
#   6. 全遮蔽行: vendor LSE=0、out=0；naive exp(score−LSE) 会产出垃圾 P，
#      mask/casual 零填充后归 0（native 语义 P=0、out=0）。
#
# A1 集成: register._fwd 以 dropout_p=0 调用本函数；dropout 在 Function 层
# 落地。direct requires_grad 建图回退 torch_level（与 p800 同——避免另建
# autograd 图）。backward 走 A1 数学梯度（无需 p800 的 fused bwd 状态）。
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


def _load_neighbor(module_name: str, path: Path):
    """Unique module name for multi-operator perf processes (#10)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:  # Package-style import
    from ..reference import _validate
    from .torch_level import sdpa_math_torch
except ImportError:  # Standalone/top-level-kernel import
    _reference_module = _load_neighbor(
        "sdpa_math_thead_reference",
        Path(__file__).resolve().parents[1] / "reference.py",
    )
    _torch_module = _load_neighbor(
        "sdpa_math_thead_torch_level",
        Path(__file__).with_name("torch_level.py"),
    )
    _validate = _reference_module._validate
    sdpa_math_torch = _torch_module.sdpa_math_torch


PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # PPU torch 经 CUDA 兼容层直曝 cuda:N
_FAST_DTYPES = (torch.float16, torch.bfloat16)
_HAS_PPU_STACK = None


def _has_ppu_stack() -> bool:
    """进程级 PPU 栈识别（T1/T2 同款双路探测; 结果缓存——栈身份进程内不变）。"""
    global _HAS_PPU_STACK
    if _HAS_PPU_STACK is None:
        try:
            from triton.runtime import driver
            if "PPU" in type(driver.active).__name__:
                _HAS_PPU_STACK = True
                return True
        except Exception:
            pass
        try:
            _HAS_PPU_STACK = torch.cuda.is_available() and any(
                "PPU" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            )
        except Exception:
            _HAS_PPU_STACK = False
    return _HAS_PPU_STACK


def _device_is_ppu(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return "PPU" in torch.cuda.get_device_name(device)
    except Exception:
        return False


def _aligned_bias(bias: torch.Tensor) -> torch.Tensor:
    """PPU efficient attn_bias 16B stride 对齐（T1 sdpa _aligned_bias 同款）。"""
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
    if enable_gqa and query.shape[1] != key.shape[1]:
        rep = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    return query, key, value


def _materialize_p(query, key, logsumexp, is_causal, scale, attn_mask=None):
    """P = exp(scale·QKᵀ − LSE)，causal/mask 位显式置 0。

    LSE 可能被厂商 pad（efficient: 32 倍数; flash: 精确 Sq）——统一切片。
    """
    sq = query.shape[-2]
    lse = logsumexp[..., :sq] if logsumexp.shape[-1] >= sq else logsumexp
    scores = torch.matmul(query, key.transpose(-1, -2))
    scores.mul_(scale if scale is not None else query.shape[-1] ** -0.5)
    scores.sub_(lse.unsqueeze(-1).to(scores.dtype)).exp_()
    if is_causal:
        skv = key.shape[-2]
        causal = torch.ones((sq, skv), dtype=torch.bool,
                            device=scores.device).tril()
        scores.masked_fill_(~causal, 0)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, 0)
        else:
            scores.masked_fill_(torch.isneginf(attn_mask), 0)
    return scores


def _fast_eligible(query, key, value, attn_mask, dropout_p, dropout_mask,
                   scale) -> bool:
    if not _has_ppu_stack() or not _device_is_ppu(query.device):
        return False
    if query.dtype not in _FAST_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if dropout_p != 0.0 or dropout_mask is not None:
        return False
    if attn_mask is not None and attn_mask.dim() != 4:
        return False
    if isinstance(scale, torch.Tensor):
        try:
            float(scale)
        except Exception:
            return False
    return True


def sdpa_math_thead_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    dropout_mask: torch.Tensor | None = None,
    *,
    scale: float | torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(output, attn_probs)`` via vendor flash/efficient + exact-P."""
    _validate(query, key, value, attn_mask, is_causal, enable_gqa)

    build_graph_directly = (
        torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )
    if not _fast_eligible(query, key, value, attn_mask, dropout_p,
                          dropout_mask, scale) or build_graph_directly:
        return sdpa_math_torch(
            query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
            scale=scale, enable_gqa=enable_gqa,
        )

    actual_scale = (
        scale if scale is not None
        else 1.0 / math.sqrt(query.shape[-1])
    )
    if isinstance(actual_scale, torch.Tensor):
        actual_scale = float(actual_scale)

    query, key, value = _expand_gqa(query, key, value, enable_gqa)

    # PPU flash 对非方阵 causal（Sq≠Skv）不生效（首测: flash O err≈2.0 vs
    # efficient/F.sdpa err≈0）——非方阵 causal 走 efficient；方阵 causal 或
    # 无 causal 的 maskless 走 flash（原生同 kernel, LSE 无 pad）。
    use_flash = (
        attn_mask is None
        and (not is_causal or query.shape[-2] == key.shape[-2])
    )
    if use_flash:
        r = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, 0.0, is_causal, False, scale=actual_scale,
        )
        out, lse = r[0], r[1]
        probs = _materialize_p(query, key, lse, is_causal, actual_scale)
        return out, probs

    # masked / 非方阵 causal → efficient 委托（flash schema 无 attn_bias）
    bias = attn_mask
    if bias is not None:
        if bias.shape[1] == 1 and query.shape[1] != 1:
            # bool_b1 等头维广播 mask——efficient 要求 bias 头维与 Hq 对齐
            bias = bias.expand(
                bias.shape[0], query.shape[1], *bias.shape[2:]).contiguous()
        if bias.dtype == torch.bool:
            src = bias
            bias = torch.zeros(src.shape, dtype=query.dtype,
                               device=src.device)
            bias.masked_fill_(~src, float("-inf"))
        elif bias.dtype != query.dtype:
            bias = bias.to(query.dtype)
        bias = _aligned_bias(bias)

    r = torch.ops.aten._scaled_dot_product_efficient_attention(
        query, key, value, attn_bias=bias, compute_log_sumexp=True,
        dropout_p=0.0, is_causal=is_causal, scale=actual_scale,
    )
    out, lse = r[0], r[1]
    probs = _materialize_p(query, key, lse, is_causal, actual_scale,
                           attn_mask=attn_mask)
    return out, probs
