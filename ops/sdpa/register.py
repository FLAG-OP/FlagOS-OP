# A1 注册: torch.library.Library("aten","IMPL") 接管
# aten::scaled_dot_product_attention。
# 平台 dispatch key 由 _profile 提供: torch_npu=AutogradPrivateUse1，
# Kunlunxin XMLIR=AutogradCUDA（绑定结论见 ../PLATFORM.md）。
# wt <wangt635@ustc.edu.cn>
#
# 实测结论（Ascend910 + torch_npu 2.10，2026-09-16）:
#   1. 仅注册 PrivateUse1 无效——torch_npu 的 C++ AutogradPrivateUse1
#      包装（RegisterAutogradNPU.cpp）不 redispatch，直接调内部实现，
#      Python 层后端 key 覆盖永远不会被走到（dispatch dump 中我们的
#      注册显示 active 但 call count=0）
#   2. 注册 AutogradPrivateUse1 两种模式（grad/no_grad）都命中，
#      是 torch_npu 栈上 A1 的正确拦截点
#   3. 朴素函数注册会破坏 autograd（输出无 grad_fn）——必须用
#      autograd.Function 包装: forward=Triton kernel，backward=标准
#      SDPA 数学梯度（fp32 ATen 组合，与 reference 同层）
import torch

try:  # Package-style import: ops.sdpa.register
    from .kernel.auto_dispatch import sdpa_auto
    from .kernel.auto_dispatch import install_patch, stats as _auto_stats
    from .kernel.triton_level import PLATFORM, sdpa_triton
except ImportError:  # Standalone import with OP_DIR on sys.path
    from kernel.auto_dispatch import sdpa_auto
    from kernel.auto_dispatch import install_patch, stats as _auto_stats
    from kernel.triton_level import PLATFORM, sdpa_triton


class _SDPA_A1_Function(torch.autograd.Function):
    """forward=Triton kernel; backward=fp32 数学梯度（无 Triton bwd）。"""

    @staticmethod
    def forward(ctx, query, key, value, attn_mask, dropout_p, is_causal,
                scale, enable_gqa):
        # wt 2026-09-22-fix impl="auto" 时走智能路由（大S无mask→原生，
        # 详见 kernel/auto_dispatch.py）；默认 "triton" 保持原验证口径
        # # wt <wangt635@ustc.edu.cn>
        if getattr(_SDPA_A1_Function, "_impl", "triton") == "auto":
            fn = sdpa_auto
        else:
            fn = sdpa_triton
        ctx.save_for_backward(query, key, value)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa
        return fn(query, key, value, attn_mask, dropout_p, is_causal,
                  scale, enable_gqa)

    @staticmethod
    def backward(ctx, do):
        # 标准 SDPA 梯度（无 dropout）:
        #   A = QKᵀ·s + B;  P = softmax(A);  O = PV
        #   dV = Pᵀ dO;  dP = dO Vᵀ;  dA = P ⊙ (dP − rowsum(dP⊙P))
        #   dQ = (dA·s) K;  dK = (dA·s)ᵀ Q;  dB = dA（加性 mask 时）
        # bool mask / causal 的遮蔽位 P=0 → dA=0，梯度自动正确
        query, key, value = ctx.saved_tensors
        B_, Hq, Sq, D = query.shape
        _, Hkv, Skv, _ = key.shape
        gqa_rep = Hq // Hkv

        q = query.float()
        k = key.float()
        v = value.float()
        do = do.float()
        s = ctx.scale if ctx.scale is not None else D ** -0.5

        if gqa_rep != 1:
            k = k.repeat_interleave(gqa_rep, dim=1)
            v = v.repeat_interleave(gqa_rep, dim=1)

        # 分步构造（与 reference 完全同路，保证语义一致）
        scores = torch.matmul(q, k.transpose(-1, -2)) * s
        if ctx.is_causal and Sq == Skv:
            causal = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool,
                                           device=q.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        if ctx.attn_mask is not None:
            if ctx.attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~ctx.attn_mask, float("-inf"))
            else:
                scores = scores + ctx.attn_mask.float()
        p = torch.softmax(scores, dim=-1)

        dv = torch.matmul(p.transpose(-1, -2), do)          # (B,H,Skv,D)
        dp = torch.matmul(do, v.transpose(-1, -2))          # (B,H,Sq,Skv)
        da = p * (dp - (dp * p).sum(dim=-1, keepdim=True))  # softmax 雅可比
        dq = torch.matmul(da, k) * s
        dk = torch.matmul(da.transpose(-1, -2), q) * s

        if gqa_rep != 1:
            dk = dk.reshape(B_, Hkv, gqa_rep, Skv, D).sum(dim=2)
            dv = dv.reshape(B_, Hkv, gqa_rep, Skv, D).sum(dim=2)

        dq = dq.to(query.dtype)
        dk = dk.to(key.dtype)
        dv = dv.to(value.dtype)

        d_mask = None
        if (ctx.attn_mask is not None
                and ctx.attn_mask.dtype != torch.bool
                and ctx.attn_mask.requires_grad):
            d_mask = da.to(ctx.attn_mask.dtype)

        return dq, dk, dv, d_mask, None, None, None, None


def _causal_or_bool_bias(ctx, query):
    return None  # 占位（分步构造已覆盖）


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                impl: str = "triton"):
    """impl: "triton"=纯自研（默认，验证口径不变） | "auto"=智能路由
    （大S无mask→厂商原生，kernel/auto_dispatch.py，见
    reports/fusion_vs_dispatch.md）"""
    """接管 aten::scaled_dot_product_attention（torch_npu 栈）。

    dispatch_key: torch_npu 栈必须用 AutogradPrivateUse1（见模块注释）；
                  其他后端（CUDA 等）可传各自 key。
    counter: 可选 dict，'n' 计数（op 层拦截验证）。
    返回 (lib, fn)，lib 必须保持引用。
    """
    # wt 2026-09-16-fix 注册守卫: torch_npu 特有 key 上来注册非 npu 平台
    # 的 kernel 是跨平台误用——在注册时就拦截，而不是运行时静默错。
    # # wt <wangt635@ustc.edu.cn>
    if dispatch_key in ("AutogradPrivateUse1", "PrivateUse1") and \
            not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError(
            f"register_a1 绑定 PLATFORM={PLATFORM!r}，dispatch_key="
            f"{dispatch_key!r} 需要 torch_npu 可用环境。跨平台集成请"
            f"按 PLATFORM.md §4 选择对应实现目录。")
    _SDPA_A1_Function._impl = impl  # 类属性: forward 内读取

    if impl == "auto":
        # wt 2026-09-22-fix auto 模式不再走 aten 注册——Python 层无法
        # 旁路 dispatcher（注册内转发原生会无限递归, RecursionError
        # 实证）, 改用函数层 patch（kernel/auto_dispatch.py install_patch）。
        # 业务代码经 F.sdpa 自动获得 shape-aware 路由。
        # # wt <wangt635@ustc.edu.cn>
        install_patch()
        if counter is not None:
            counter["patched"] = True
        return _auto_stats  # 返回 stats 供调用方断言（无 lib 需保持引用）

    # impl="triton": 原 aten 注册路径（验证口径不变）。
    # CUDA/XMLIR dispatcher 可能省略 schema 默认值，必须补齐后再进入
    # autograd.Function；torch_npu 此前实测由 dispatcher 展开全部参数。
    def impl_fn_(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
        if counter is not None:
            counter["n"] += 1
        return _SDPA_A1_Function.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale,
            enable_gqa)

    impl_fn = impl_fn_

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("scaled_dot_product_attention", impl_fn, dispatch_key)
    return lib


def unhook_a1(lib) -> None:
    """torch.library 无官方撤销 API: 删除对象后注册仍在（实测 torch 2.10）。
    "恢复原生"验证须在未注册的独立进程进行（test/op_level.py 的做法:
    注册前先缓存原生输出，撤销断言改为跨进程对照）。"""
    del lib
