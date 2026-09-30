#!/usr/bin/env python3
"""native 语义证据脚本——reference / register / 三层测试注释里引用的
"probes/native_semantics.py" 就是本文件（独立进程运行，勿与注册混跑）。

运行:
  python3 probes/native_semantics.py [ascend910|cpu]

覆盖的实测结论（每节打印证据表）:
  1. schema 与注册面: aten::_scaled_dot_product_attention_math 是
     CompositeImplicitAutograd，任意 backend key 的 Python impl 可覆盖；
  2. F.sdpa 参数到达形态: bool mask → float32 加性 -inf、dropout_mask
     恒为 None、scale/enable_gqa 走 kwargs；NPU 上 F.sdpa 走融合注意力、
     从不进本算子（NPU 消费方只能直调 torch.ops）；
  3. dropout 双规则表（native 最容易抄错的语义）:
       · 显式 dropout_mask: keep=(mask!=0)（幅值忽略）、p=0 时掩码完全
         被忽略、返回 P=P0⊙keep（**不缩放**）、O=(P0⊙keep/(1-p))@V
         → O ≠ P@V；
       · 随机路径（F.sdpa 唯一形态）: P=P0⊙keep/(1-p)、O=P@V（自洽）；
  4. bool attn_mask **直调**怪癖: 按 0/1 加性（非 -inf 遮蔽）——与
     F.sdpa 路径不一致，故 reference 有意偏离直调、对齐 F.sdpa；
  5. 冲突与边界: causal+mask 报错（bool/float 皆然）、全 -inf 行 → 0；
  6. 注册键覆盖: 只注册 Autograd* 时 inference_mode 不命中；补注册
     纯设备 key 后命中（register.py 成对注册的原因）。
"""
from __future__ import annotations

import sys
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_DIR))


def _head(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def _raw_register(key: str, fn):
    """单 key 注册（register_a1 是成对注册，这里要单独验证每个键）。"""
    import torch
    lib = torch.library.Library("aten", "IMPL")
    lib.impl("_scaled_dot_product_attention_math", fn, key)
    return lib


def sec_schema():
    import torch
    _head("1) schema 与注册面")
    try:
        print(torch._C._get_schema(
            "aten::_scaled_dot_product_attention_math", ""))
    except Exception:
        print(list(torch.ops.aten._scaled_dot_product_attention_math._schemas))
    try:  # 2.10: 各 backend key 上登记的 kernel
        dump = torch._C._dispatch_dump_table(
            "aten::_scaled_dot_product_attention_math")
        lines = [ln for ln in dump.splitlines()
                 if "CompositeImplicitAutograd" in ln or "math" in ln.lower()]
        print("\n".join(lines[:6]))
        assert "CompositeImplicitAutograd" in dump, "非 composite?"
    except Exception as e:  # noqa: BLE001
        print(f"  (dispatch dump 不可用: {e})")


def sec_fsdp_params(profile):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from kernel.torch_level import sdpa_math_torch

    _head("5) F.sdpa 到达本算子时的参数形态（登记一个记录型 impl）")
    dev = profile.torch_device
    rec: dict = {}

    def recorder(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                 dropout_mask=None, scale=None, enable_gqa=False):
        m = attn_mask
        rec["attn_mask"] = (
            "None" if m is None else
            f"dtype={m.dtype} shape={tuple(m.shape)} "
            f"range=[{float(m.min()):.1f},{float(m.max()):.1f}]")
        rec.update(dropout_p=dropout_p, dropout_mask=dropout_mask,
                   is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        return sdpa_math_torch(q, k, v, m, dropout_p, is_causal, dropout_mask,
                               scale=scale, enable_gqa=enable_gqa)

    # CPU 键: F.sdpa(MATH) 在 CPU 上会真正走到本算子
    lib = _raw_register("CPU", recorder)
    g = torch.Generator("cpu").manual_seed(0)
    qc = torch.randn(1, 4, 8, 16, generator=g)
    kc = torch.randn(1, 4, 8, 16, generator=g)
    vc = torch.randn(1, 4, 8, 16, generator=g)
    bool_mask = torch.rand(1, 4, 8, 8, generator=g) > 0.5

    rec.clear()
    with sdpa_kernel(SDPBackend.MATH):
        F.scaled_dot_product_attention(qc, kc, vc, attn_mask=bool_mask,
                                       dropout_p=0.2, scale=0.5)
    print("  F.sdpa(attn_mask=bool, dropout_p=0.2, scale=0.5) → 记录:")
    for kk, vv in rec.items():
        print(f"    {kk:12s} = {vv}")
    assert rec["attn_mask"].startswith("dtype=torch.float32")
    assert "-inf" in rec["attn_mask"] or "-3.4e+38" in rec["attn_mask"] \
        or "-inf" in rec["attn_mask"]
    assert rec["dropout_mask"] is None, "F.sdpa 必须传 dropout_mask=None"
    assert rec["dropout_p"] == 0.2 and rec["scale"] == 0.5

    rec.clear()
    with sdpa_kernel(SDPBackend.MATH):
        F.scaled_dot_product_attention(qc, kc, vc, is_causal=True)
    print(f"  F.sdpa(is_causal=True) → is_causal={rec['is_causal']}, "
          f"attn_mask={rec['attn_mask']}")
    del lib  # 单 key 注册不可撤销（进程退出才生效）

    # NPU: F.sdpa 被 torch_npu 路由到融合注意力 —— 本算子不被调用
    if profile.vendor == "ascend":
        import torch as t
        n0 = {"n": 0}

        def count_only(*a, **k):  # noqa: ANN002, ANN003
            n0["n"] += 1
            return sdpa_math_torch(*a, **k)

        lib2 = _raw_register("PrivateUse1", count_only)
        qn = qc.to(dev)
        kn = kc.to(dev)
        vn = vc.to(dev)
        with sdpa_kernel(SDPBackend.MATH):
            F.scaled_dot_product_attention(qn, kn, vn, is_causal=True)
        print(f"  NPU 上 F.sdpa(MATH) 命中本算子次数 = {n0['n']} "
              f"（0 = torch_npu 走融合注意力，直调 torch.ops 才是消费方）")
        del lib2


def sec_dropout_table(profile):
    import torch

    from reference import make_dropout_mask

    _head("2) dropout 双规则表（native 实测，参考实现照抄口径）")
    dev = profile.torch_device
    fn = _op(profile)

    B, H, S, D = 1, 2, 8, 16
    g = torch.Generator("cpu").manual_seed(0)
    q = (torch.randn(B, H, S, D, generator=g)).to(dev)
    k = (torch.randn(B, H, S, D, generator=g)).to(dev)
    v = (torch.randn(B, H, S, D, generator=g)).to(dev)

    o0, p0 = fn(q, k, v, None, 0.0, False, None)          # (out, probs)
    rows = []

    def add(tag, P, O, expect_p, expect_o):
        fp = float("nan") if expect_p is None else \
            (P - expect_p).abs().max().item()
        fo = float("nan") if expect_o is None else \
            (O - expect_o).abs().max().item()
        rows.append((tag, fp, fo))

    keep1 = torch.ones(B, H, S, S, device=dev)
    keep0 = torch.zeros(B, H, S, S, device=dev)
    mixed = torch.ones(B, H, S, S, device=dev)
    mixed[..., S // 2:] = 0.0
    mixed[0, 0, 0, 0] = 7.0        # 非 0 幅值必须被忽略 → 仍 keep
    mixed[0, 0, 1, 1] = -3.0       # 非 0 负值同样 keep

    # 显式掩码: keep=(mask!=0)，返回 P 不缩放，O 再除 (1-p)
    for tag, mk, p in [("explicit all-keep p=0.5", keep1, 0.5),
                       ("explicit all-drop p=0.5", keep0, 0.5),
                       ("explicit mixed p=0.5", mixed, 0.5),
                       ("explicit mixed p=0 (掩码忽略)", mixed, 0.0)]:
        O, P = fn(q, k, v, None, p, False, mk)
        if p == 0.0:
            add(tag, P, O, p0, o0)
            continue
        keep = (mk != 0).to(p0.dtype)
        add(tag, P, O, p0 * keep,
            torch.matmul(p0 * keep / (1 - p), v))

    # 随机路径: P 与 O 同乘 keep/(1-p)，O = P@V（自洽）
    O, P = fn(q, k, v, None, 0.5, False, None)
    with torch.no_grad():
        keep = (P != 0).to(p0.dtype)
    add("random p=0.5 (O=P@V 自洽)", P, O, None,
        torch.matmul(P, v))
    kept = P != 0                      # 只在 keep 位看幅值比例
    ratio = P[kept] / p0[kept]
    keep_rate = float(kept.float().mean())

    print(f"  {'case':34s} {'|P−expected|':>14s} {'|O−expected|':>14s}")
    print("  " + "-" * 66)
    for tag, fp, fo in rows:
        fs = "   (see below)" if fp != fp else f"{fp:14.2e}"
        print(f"  {tag:34s} {fs} {fo:14.2e}")
    print(f"  随机路径: keep率={keep_rate:.3f}(期望 0.5), "
          f"kept 上 P/P0 中位数={float(ratio.median()):.3f}(期望 2.0), "
          f"O=P@V 已在上表末行自洽")
    print("  结论: 显式掩码 P 不缩放、O 缩放 1/(1-p) → O≠P@V；"
          "随机路径两者同缩放 → O=P@V")


def _op(profile):
    """未注册进程里 = native（直调）。"""
    import torch

    def fn(q, k, v, m=None, p=0.0, causal=False, dm=None, **kw):
        return torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, m, p, causal, dm, **kw)
    return fn


def sec_bool_quirk(profile):
    import torch

    _head("3) bool attn_mask 直调怪癖（reference 的唯一有意分歧）")
    dev = profile.torch_device
    fn = _op(profile)
    B, H, S, D = 1, 2, 8, 16
    g = torch.Generator("cpu").manual_seed(1)
    q = torch.randn(B, H, S, D, generator=g).to(dev)
    k = torch.randn(B, H, S, D, generator=g).to(dev)
    v = torch.randn(B, H, S, D, generator=g).to(dev)
    mask = torch.rand(B, H, S, S, generator=g) > 0.5
    mask = mask.to(dev)

    _, P = fn(q, k, v, mask, 0.0, False, None)   # (out, probs)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (D ** 0.5)
    add_01 = torch.softmax(scores + mask.float(), dim=-1)      # 0/1 加性
    fill_inf = torch.nan_to_num(
        torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1),
        nan=0.0)                                              # -inf 遮蔽
    e_add = (P.float() - add_01).abs().max().item()
    e_fill = (P.float() - fill_inf).abs().max().item()
    print(f"  直调 vs 0/1 加性解释: {e_add:.2e}")
    print(f"  直调 vs -inf 遮蔽解释: {e_fill:.2e}")
    print("  结论: 直调=0/1 加性怪癖（e_add 小）；F.sdpa 路径先转 float32 "
          "-inf 加性 → 到达时是遮蔽语义。"
          "reference 按遮蔽实现（对齐 F.sdpa），仅与直调在 bool mask 下有意不同。")


def sec_edges(profile):
    import torch

    _head("4) 冲突与边界")
    dev = profile.torch_device
    fn = _op(profile)
    B, H, S, D = 1, 2, 8, 16
    g = torch.Generator("cpu").manual_seed(2)
    q = torch.randn(B, H, S, D, generator=g).to(dev)
    k = torch.randn(B, H, S, D, generator=g).to(dev)
    v = torch.randn(B, H, S, D, generator=g).to(dev)

    for tag, mk in [("bool", torch.rand(B, H, S, S, generator=g) > 0.5),
                    ("float", torch.randn(B, H, S, S, generator=g) * 0.1)]:
        try:
            fn(q, k, v, mk.to(dev), 0.0, True, None)
            print(f"  causal + {tag} mask: 未报错（意外）")
        except RuntimeError as e:
            print(f"  causal + {tag} mask → RuntimeError: {str(e)[:72]}")

    # (a) bool 全 False 直调 → 走 0/1 加性怪癖（等价于不遮蔽），不是 0
    bad = torch.zeros(B, H, S, S, dtype=torch.bool, device=dev)
    _, Pb = fn(q, k, v, bad, 0.0, False, None)
    _, p_ref = fn(q, k, v, None, 0.0, False, None)
    print(f"  bool 全 False 直调: max|P−P(no mask)|="
          f"{(Pb.float() - p_ref.float()).abs().max().item():.2e} "
          f"（0 = 加性怪癖，没有遮蔽任何位置）")
    # (b) float mask 一整行 -inf → 该行 P=0、O 有限（native 遮蔽语义）
    fm = torch.zeros(B, H, S, S, device=dev)
    fm[..., 0, :] = float("-inf")
    O, P = fn(q, k, v, fm, 0.0, False, None)
    print(f"  float mask 整行 -inf: 该行 max|P|={P[..., 0, :].abs().max():.2e}, "
          f"max|O|={O.abs().max().item():.2e}, "
          f"finite={bool(torch.isfinite(P).all() and torch.isfinite(O).all())}"
          f"（native: 全 -inf 行 → 0，不是 NaN）")


def sec_keys(profile):
    import torch

    _head("6) 注册键覆盖（register.py 成对注册的原因）")
    dev = profile.torch_device
    plain = ("PrivateUse1" if dev.startswith("npu") else "CPU")
    auto = ("AutogradPrivateUse1" if dev.startswith("npu") else "AutogradCPU")
    rec = {"n": 0}

    def impl(*a, **k):  # noqa: ANN002, ANN003
        rec["n"] += 1
        from kernel.torch_level import sdpa_math_torch
        return sdpa_math_torch(*a, **k)

    libs = [_raw_register(auto, impl)]   # 第一步: 只注册 Autograd*
    g = torch.Generator("cpu").manual_seed(3)
    q = torch.randn(1, 2, 8, 16, generator=g).to(dev).requires_grad_(True)
    k = torch.randn(1, 2, 8, 16, generator=g).to(dev).requires_grad_(True)
    v = torch.randn(1, 2, 8, 16, generator=g).to(dev).requires_grad_(True)

    n0 = rec["n"]
    out, _ = torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, None, 0.0, True, None)
    grad_hit = rec["n"] > n0
    n0 = rec["n"]
    with torch.inference_mode():
        torch.ops.aten._scaled_dot_product_attention_math(
            q.detach(), k.detach(), v.detach(), None, 0.0, True, None)
    inf_hit_auto = rec["n"] > n0
    print(f"  只注册 {auto}: grad 路径命中={grad_hit}, "
          f"grad_fn={out.grad_fn is not None}, "
          f"inference_mode 命中={inf_hit_auto}  ← 此处为 False")

    libs.append(_raw_register(plain, impl))   # 第二步: 补注册纯设备键
    n0 = rec["n"]
    with torch.inference_mode():
        torch.ops.aten._scaled_dot_product_attention_math(
            q.detach(), k.detach(), v.detach(), None, 0.0, True, None)
    inf_hit_both = rec["n"] > n0
    print(f"  再注册 {plain}: inference_mode 命中={inf_hit_both}")
    print("  结论: Autograd* 单键覆盖不了 inference_mode（且反向会报 "
          "'an autograd kernel was not registered' 警告）→ 必须成对注册。")
    print(f"  (lib 引用保留在本进程: {len(libs)} 个)")


def main() -> int:
    from _profile import load_profile
    profile = load_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[native_semantics] {profile.summary()}")
    # 顺序: 先跑依赖"未注册原生直调"的节（3/4/5），再跑会注册 impl 的
    # 节（2/6）——注册不可撤销
    sec_schema()
    sec_dropout_table(profile)
    sec_bool_quirk(profile)
    sec_edges(profile)
    sec_fsdp_params(profile)
    sec_keys(profile)
    print("\n全部证据节完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
