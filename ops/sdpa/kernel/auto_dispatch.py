# 智能路由（截流下发）: shape-aware dispatch 自研 Triton ↔ 厂商原生。
# wt <wangt635@ustc.edu.cn>
#
# 依据（reports/fusion_vs_dispatch.md 实测）:
#   - S≥1024 无 mask 时 Triton 栈差距 6-12x → 路由原生（CANN 闪电注意力）
#   - 小 S / 带自定义 mask / GQA 组合 → 自研 Triton（长尾兜底）
#   - 大 S 段拿到 98-99% 最优
#
# 实现形式（关键工程决策, 2026-09-22 实测教训）:
#   ✗ aten 注册 + 内部转发原生: Python 层无法旁路 dispatcher——
#     torch.ops.aten 调用恒走完整 dispatch（含我们的 AutogradPrivateUse1
#     注册）→ 无限递归（RecursionError 实证, 见 test/op_level_auto.py
#     调试记录）
#   ✓ monkey-patch 函数层: 保存原始 F.scaled_dot_product_attention 引用,
#     大 S 直调原始引用 → torch_npu 原生（无注册干扰）; Python 业务代码
#     （vLLM/transformers 均经 F.sdpa）零改动受益
#   - register_a1(impl="auto") 内部即调用 install_patch（不注册 aten）
#
# 开关（环境变量, 进程级）:
#   SDPA_DISPATCH_MODE = auto（默认）| triton（强制自研）| native（强制原生）
#   SDPA_DISPATCH_S    = 路由阈值, 默认 1024（ascend910 实测拐点）
#
# ⚠ 平台绑定: 阈值 1024 与"mask=None 才路由原生"均来自 ascend910 实测
# （native 拒绝 causal+mask 并存, 见 kernel/_native_shim.py）。移植/换栈
# 后须按 MERGE.md Step2 重验并重新标定（PLATFORM.md §2）。
from __future__ import annotations

import os

import torch
import torch.nn.functional as _F

_ORIG_SDPA = _F.scaled_dot_product_attention     # 原始引用（patch 前固定）
_patched = False

# 路由统计（op 层断言与可观测性用; 进程级累计）
stats = {"routed_triton": 0, "routed_native": 0, "forced_triton": 0,
         "forced_native": 0}


def _mode() -> str:
    return os.environ.get("SDPA_DISPATCH_MODE", "auto").lower()


def _threshold() -> int:
    return int(os.environ.get("SDPA_DISPATCH_S", "1024"))


def reset_stats() -> None:
    for k in stats:
        stats[k] = 0


def _triton_with_grad(query, key, value, attn_mask, dropout_p, is_causal,
                      scale, enable_gqa):
    """自研 Triton 前向 + 统一数学梯度（复用 register 的 autograd 包装）。"""
    try:  # Package-style import: ops.sdpa.kernel.auto_dispatch
        from .. import register
    except ImportError:
        import register
    saved = register._SDPA_A1_Function._impl
    register._SDPA_A1_Function._impl = "triton"   # 防 auto 循环
    try:
        return register._SDPA_A1_Function.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa)
    finally:
        register._SDPA_A1_Function._impl = saved


def sdpa_auto(query, key, value, attn_mask=None, dropout_p=0.0,
              is_causal=False, scale=None, enable_gqa=False):
    """shape-aware SDPA: 大 S 无 mask → 原始 F.sdpa（厂商原生后端）;
    其余 → 自研 Triton（带 autograd）。"""
    mode = _mode()
    if mode == "triton":
        stats["forced_triton"] += 1
        return _triton_with_grad(query, key, value, attn_mask, dropout_p,
                                 is_causal, scale, enable_gqa)
    if mode == "native":
        stats["forced_native"] += 1
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, enable_gqa=enable_gqa)

    S = query.shape[-2]
    if S >= _threshold() and attn_mask is None and dropout_p == 0.0:
        stats["routed_native"] += 1
        # 原始引用 → 未注册路径 → torch_npu 原生（自带 autograd）
        return _ORIG_SDPA(query, key, value, attn_mask=None,
                          dropout_p=0.0, is_causal=is_causal, scale=scale,
                          enable_gqa=enable_gqa)
    stats["routed_triton"] += 1
    return _triton_with_grad(query, key, value, attn_mask, dropout_p,
                             is_causal, scale, enable_gqa)


def install_patch() -> None:
    """把 F.scaled_dot_product_attention 替换为 sdpa_auto（幂等）。"""
    global _patched
    if _patched:
        return
    _F.scaled_dot_product_attention = sdpa_auto
    # torch.nn.functional 的 __all__/模块属性直接生效
    _patched = True


def remove_patch() -> None:
    """还原原始 F.sdpa（幂等）。"""
    global _patched
    if not _patched:
        return
    _F.scaled_dot_product_attention = _ORIG_SDPA
    _patched = False
