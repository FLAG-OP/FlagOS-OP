# A1 注册: torch.library.Library("aten","IMPL") 接管
# aten::_scaled_dot_product_attention_math（SDPA 的 math 后端）。
#
# 本算子是 CompositeImplicitAutograd（torch 2.10 dispatch dump 实测:
# 所有 backend key 上登记的都是同一个 math composite）。因此:
#   1. 任意 backend key 的 Python impl 都能覆盖 composite（CPU /
#      AutogradCPU / PrivateUse1 / AutogradPrivateUse1 实测均可命中）；
#   2. F.sdpa 的 MATH 后端在 CPU 上会真正走到这里（op 层拦截证据），
#      而 torch_npu 把 F.sdpa 整体路由到 npu_fusion_attention、**从不
#      调用**本算子——NPU 上的消费方只能直调 torch.ops（framework 层
#      据此设计，证据 probes/native_semantics.py）。
# 3. 朴素函数注册会破坏 autograd（forward 含 Triton launch，输出无
#    grad_fn）——必须用 autograd.Function: forward=自研实现，
#    backward=fp32 数学梯度（含第二输出 attn_probs 的梯度）。
#
# wt <wangt635@ustc.edu.cn>
from __future__ import annotations

import torch

try:  # Package-style import: ops.sdpa_math.register
    from .kernel.torch_level import sdpa_math_torch
    from .kernel.triton_level import PLATFORM, sdpa_math_triton
    from .reference import _softmax01, make_dropout_mask, sdpa_math_reference
except ImportError:  # Standalone import with OP_DIR on sys.path
    from kernel.torch_level import sdpa_math_torch
    from kernel.triton_level import PLATFORM, sdpa_math_triton
    from reference import _softmax01, make_dropout_mask, sdpa_math_reference

_IMPLS = {
    "triton": sdpa_math_triton,
    "torch": sdpa_math_torch,
    "reference": sdpa_math_reference,
}


def _fwd(impl, ctx, query, key, value, attn_mask, dropout_p, is_causal,
         dropout_mask, scale, enable_gqa):
    """autograd.Function 前向（impl 由调用方绑定，见 register_a1）。

    抽成模块级函数的原因: 一个进程常注册多个 key（npu=triton + CPU=torch
    供 gradcheck/F.sdpa），若 forward 读共享类属性会互相覆盖。
    """
    fn = _IMPLS[impl]
    out, probs = fn(query, key, value, attn_mask, 0.0, is_causal, None,
                    scale=scale, enable_gqa=enable_gqa)

    keep = None
    if dropout_p > 0.0:
        Hq, Hkv = query.shape[1], value.shape[1]
        vv = (value.repeat_interleave(Hq // Hkv, dim=1)
              if (Hq != Hkv and enable_gqa) else value)
        if dropout_mask is None:
            # 随机路径（native）: P 与 O 同乘 keep/(1-p)，自洽
            keep = (torch.rand(probs.shape, device=probs.device)
                    >= dropout_p).to(probs.dtype)
            probs = probs * (keep / (1.0 - dropout_p))
            out = torch.matmul(probs, vv)
        else:
            # 显式路径（native）: keep=(mask!=0)，返回的 P 不缩放、O 缩放
            keep = (dropout_mask != 0).to(probs.dtype).broadcast_to(
                probs.shape)
            probs = probs * keep
            out = torch.matmul(probs / (1.0 - dropout_p), vv)

    ctx.save_for_backward(query, key, value)
    ctx.attn_mask = attn_mask
    ctx.keep = keep                 # None 或 0/1 张量（随机路径在此自持）
    ctx.dropout_p = dropout_p
    ctx.drop_explicit = dropout_mask is not None
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.enable_gqa = enable_gqa
    return out, probs


class _SDPAMath_A1_Function(torch.autograd.Function):
    """forward=自研实现（triton/torch）; backward=fp32 数学梯度。

    算子有两个输出 (out, attn_probs)，backward 因此接收两份梯度:
    do    → 经 O = (P0⊙a_out)@V 回传（dV = (P0⊙a_out)ᵀ·do）
    dprobs→ 作用在返回的概率图（P0⊙a_ret）上（蒸馏/熵正则类消费方会用）
    两路在 softmax 雅可比处汇合（雅可比用**预 dropout** 的 P0）。

    dropout 统一在本层落地（impl 一律以 dropout_p=0 调用），原因:
      · 随机 keep 掩码在此生成一次 → 反向复用同一张（完全确定）;
      · native 的两条 dropout 规则（显式/随机，见 reference 顶注）只需在
        这里实现一遍，不会与三个 kernel 层实现各写一份。
    代价: 经注册路径的 dropout 调用，O=P@V 走 ATen 而非 fused triton PV
    （仅 dropout_p>0 时发生，dropout 不是本算子的性能路径）。
    """

    @staticmethod
    def forward(ctx, query, key, value, attn_mask, dropout_p, is_causal,
                dropout_mask, scale, enable_gqa):
        impl = getattr(_SDPAMath_A1_Function, "_impl", "triton")
        return _fwd(impl, ctx, query, key, value, attn_mask, dropout_p,
                    is_causal, dropout_mask, scale, enable_gqa)

    @staticmethod
    def backward(ctx, dout, dprobs):
        # 数学梯度（fp32，与 reference 同层）:
        #   S = scale·QKᵀ(+mask);  P0 = softmax(S)
        #   前向: 返回 P = P0⊙a_ret ;  O = (P0⊙a_out)@V
        #     随机路径 a_out = a_ret = keep/(1-p)
        #     显式路径 a_out = keep/(1-p), a_ret = keep      ← native 两条规则
        #   dV = (P0⊙a_out)ᵀ·dO
        #   dP0 = (dO·Vᵀ)⊙a_out + dprobs⊙a_ret
        #   dS = P0 ⊙ (dP0 − rowsum(dP0⊙P0))     softmax 雅可比（用 P0）
        #   dQ = scale·dS·K ;  dK = scale·dSᵀ·Q ;  dmask = dS（可微 float mask）
        # bool mask / causal 的遮蔽位 P0=0 → dS=0，梯度自动正确;
        # dropout 的 keep 是 0/1 指示（不可微）→ 对 dropout_mask 返回 None
        query, key, value = ctx.saved_tensors
        mask = ctx.attn_mask
        keep = ctx.keep

        B, Hq, Sq, D = query.shape
        _, Hkv, Skv, _ = key.shape
        gqa = Hq // Hkv

        # fp64 输入用 fp64 累积（gradcheck 需要），其余 fp32（与 native
        # "内部全程 fp32"一致）
        acc = torch.float64 if query.dtype == torch.float64 else torch.float32
        q = query.to(acc)
        k = key.to(acc)
        v = value.to(acc)
        if gqa != 1:
            k = k.repeat_interleave(gqa, dim=1)
            v = v.repeat_interleave(gqa, dim=1)

        s = ctx.scale if ctx.scale is not None else D ** -0.5
        if isinstance(s, torch.Tensor):
            s = float(s)

        scores = torch.matmul(q, k.transpose(-1, -2)) * s
        if ctx.is_causal:
            causal = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool,
                                           device=q.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask.to(acc)
        p = _softmax01(scores)                       # P0（预 dropout）
        if keep is None:
            a_out = a_ret = None                     # 无 dropout: 乘子 1
            pd_out = p                               # 前向乘 V 的概率
        else:
            inv = keep / (1.0 - ctx.dropout_p)       # keep/(1-p)
            a_out = inv
            a_ret = inv if not ctx.drop_explicit else keep
            pd_out = p * inv                         # 前向乘 V 的概率

        dO = dout.to(acc) if dout is not None else None
        dPd = dprobs.to(acc) if dprobs is not None else None

        dv = (torch.matmul(pd_out.transpose(-1, -2), dO) if dO is not None
              else torch.zeros_like(v))
        dP0 = torch.zeros_like(p)
        if dO is not None:
            g = torch.matmul(dO, v.transpose(-1, -2))     # dO·Vᵀ
            dP0 = g if a_out is None else g * a_out
        if dPd is not None:
            dP0 = dP0 + (dPd if a_ret is None else dPd * a_ret)

        da = p * (dP0 - (dP0 * p).sum(dim=-1, keepdim=True))  # dS
        dq = torch.matmul(da, k) * s
        dk = torch.matmul(da.transpose(-1, -2), q) * s

        if gqa != 1:
            dk = dk.reshape(B, Hkv, gqa, Skv, D).sum(dim=2)
            dv = dv.reshape(B, Hkv, gqa, Skv, D).sum(dim=2)

        d_mask = None
        if (mask is not None and mask.dtype != torch.bool
                and mask.requires_grad):
            d_mask = da.to(mask.dtype)

        return (dq.to(query.dtype), dk.to(key.dtype), dv.to(value.dtype),
                d_mask, None, None, None, None, None)


def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                impl: str | None = None):
    """接管 aten::_scaled_dot_product_attention_math（A1 路线）。

    dispatch_key: 平台 profile 提供（ascend910=AutogradPrivateUse1，
                  cpu=CPU；实测四个 key 均可覆盖该 composite）。
                  实际注册**成对的两个 key**（Autograd* + 纯设备 key），
                  原因（probes/native_semantics.py §6 实测）:
                    * 只注册 Autograd* → torch.inference_mode() 下不命中
                      （counter 不增），且反向时 torch 报
                      "an autograd kernel was not registered" 警告；
                    * 只注册纯设备 key → requires_grad 的调用先落
                      Autograd*（那里仍是原生 composite，本实现不被调用）
                      → 两键缺一不可。
    counter:      可选 dict，'n' 计数（op 层拦截验证）。
    impl:         "triton"（自研设备码，默认）| "torch"（ATen 组合）
                  | "reference"。缺省按环境推断。
    返回 lib 列表，调用方必须保持引用（否则注册被回收）。
    """
    _PAIRS = {
        "AutogradPrivateUse1": "PrivateUse1",
        "AutogradCPU": "CPU",
        "AutogradCUDA": "CUDA",
        "PrivateUse1": "AutogradPrivateUse1",
        "CPU": "AutogradCPU",
        "CUDA": "AutogradCUDA",
    }
    if dispatch_key in ("AutogradPrivateUse1", "PrivateUse1") and \
            not ((hasattr(torch, "npu") and torch.npu.is_available())
                 or (hasattr(torch, "mlu") and torch.mlu.is_available())):
        raise RuntimeError(
            f"register_a1 绑定 PLATFORM={PLATFORM!r}，dispatch_key="
            f"{dispatch_key!r} 需要 torch_npu 或 torch_mlu 可用环境。"
            f"CPU 侧请传 dispatch_key='CPU'（profile: cpu）。")
    if impl is None:
        impl = "triton" if (hasattr(torch, "npu")
                            and torch.npu.is_available()) else "torch"
    if impl not in _IMPLS:
        raise ValueError(f"未知 impl: {impl}（可选 {sorted(_IMPLS)}）")
    # 每次注册持一份 impl: 一个进程里常同时注册多个 key（如 npu 用 triton、
    # CPU 用 torch 供 gradcheck/F.sdpa 消费），共享类属性会互相覆盖。
    class _Fn(_SDPAMath_A1_Function):
        @staticmethod
        def forward(ctx, query, key, value, attn_mask, dropout_p,
                    is_causal, dropout_mask, scale, enable_gqa):
            return _fwd(impl, ctx, query, key, value, attn_mask, dropout_p,
                        is_causal, dropout_mask, scale, enable_gqa)

    keys = [dispatch_key]
    if dispatch_key in _PAIRS:
        keys.append(_PAIRS[dispatch_key])

    # dispatcher 可能省略 schema 默认值 → wrapper 先补齐再进 autograd.Function
    def impl_fn_(query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, dropout_mask=None, scale=None,
                 enable_gqa=False):
        if counter is not None:
            counter["n"] += 1
        fn = _IMPLS[impl]
        if not torch.is_grad_enabled():
            # no_grad / inference_mode: 不建图，绕开 autograd.Function
            # （inference 张量不能 save_for_backward）
            return fn(query, key, value, attn_mask, dropout_p, is_causal,
                      dropout_mask, scale=scale, enable_gqa=enable_gqa)
        return _Fn.apply(
            query, key, value, attn_mask, dropout_p, is_causal, dropout_mask,
            scale, enable_gqa)

    libs = []
    for k in keys:
        lib = torch.library.Library("aten", "IMPL")
        lib.impl("_scaled_dot_product_attention_math", impl_fn_, k)
        libs.append(lib)
    return libs


def unhook_a1(libs) -> None:
    """torch.library 无官方撤销 API: 删除对象后注册仍在（实测 torch 2.10）。
    "恢复原生"验证须在注册前采集基线，或在未注册的独立进程进行。"""
    if isinstance(libs, list):
        for x in libs:
            del x
    del libs
