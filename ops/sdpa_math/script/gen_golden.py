#!/usr/bin/env python3
"""按 goldendata/inputs_spec.yaml 生成黄金输入与参考输出（CPU fp32 权威）。

产物: goldendata/data/<case>.pt（inputs + expected + 元数据）+ index.json（sha256）。

三重交叉互验（防"参考恰好也是被测实现"式掩盖）:
  ① out   vs CPU `F.scaled_dot_product_attention`(MATH)     —— 官方调用路径
  ② out/P vs CPU native 直调 `torch.ops.aten._scaled_..._math` —— 被测算子本体
     （bool attn_mask 跳过: native 直调按 0/1 加性怪癖处理，见 reference 顶注）
   ③ 自洽: out == P @ V（显式 dropout 时按 native 怪癖 O == (P/(1-p))@V）、
      P 行和 ∈ {1, 0(全遮蔽行)}（dropout case 跳过行和——掩码会削减）
任何分歧都直接 assert——黄金生成阶段就把语义理解钉死。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parents[1]

MASK_KINDS = {"bool", "float", "bool2d", "float2d", "bool_b1"}


def _load(path, entry):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_")[:-3], HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


def _special(t: torch.Tensor, kind: str) -> None:
    if kind == "zeros_row":
        t[..., 0, :] = 0.0                 # q 全零行（P 仍应行和=1）
    elif kind == "large_row":
        t[..., 1, :] = 40.0                # 大值行（softmax 饱和稳定性）


def _gen_mask(kind, B, Hq, S, dtype, g):
    """返回 (mask, 用于 reference/kernel 的形态)。"""
    Sq, Skv = S
    if kind is None or kind == "null":
        return None
    if kind == "bool":
        return torch.rand(B, Hq, Sq, Skv, generator=g) > 0.3
    if kind == "float":
        return torch.randn(B, Hq, Sq, Skv, generator=g) * 0.1
    if kind == "bool2d":                     # (Sq,Skv) 右对齐广播
        return torch.rand(Sq, Skv, generator=g) > 0.3
    if kind == "float2d":
        return torch.randn(Sq, Skv, generator=g) * 0.1
    if kind == "bool_b1":                    # (B,1,Sq,Skv) 头维广播
        return torch.rand(B, 1, Sq, Skv, generator=g) > 0.3
    raise ValueError(f"未知 mask kind: {kind}")


def _to_dtype(m, dt):
    if m is None or m.dtype == torch.bool:
        return m
    return m.to(dt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")     # 黄金一律 CPU 生成
    ap.add_argument("--out", default=str(HERE / "goldendata"))
    args = ap.parse_args()

    ref_fn = _load("reference.py", "sdpa_math_reference")
    spec = yaml.safe_load((HERE / "goldendata/inputs_spec.yaml").read_text())
    data_dir = Path(args.out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.glob("*.pt"):          # 规格变了就清掉旧黄金
        old.unlink()

    F = torch.nn.functional
    index = {"op": spec["op"], "device": "cpu", "files": []}
    n = 0
    skipped_conflict = 0
    for case in spec["cases"]:
        S_list = case.get("S")
        if S_list is None:
            S_list = list(zip(case["Sq"], case["Skv"] * len(case["Sq"])
                              if len(case["Skv"]) == 1 else case["Skv"]))
        for dt_name in case["dtypes"]:
            dt = getattr(torch, dt_name)
            hkv_list = (case["Hkv"] if len(case["Hkv"]) == len(case["Hq"])
                        else [case["Hkv"][0]] * len(case["Hq"]))
            combos = list(itertools.product(case["B"], case["Hq"], hkv_list,
                                            S_list, case["D"],
                                            case["causal"], case["mask"],
                                            case.get("dropout_p", [0.0])))
            for (B, Hq, Hkv, S, D, causal, mask_kind, drop_p) in combos:
                Sq, Skv = S if isinstance(S, tuple) else (S, S)
                # causal + 显式 mask 是 schema 冲突（native 与本实现都报错）
                if causal and mask_kind not in (None, "null"):
                    skipped_conflict += 1
                    continue
                for k in range(case["seeds_per_case"]):
                    seed = (spec["seed_base"] + 264
                            if case.get("special") else spec["seed_base"] + n)
                    g = torch.Generator().manual_seed(seed)
                    q = torch.randn(B, Hq, Sq, D, generator=g) * case["scale"]
                    k_ = torch.randn(B, Hkv, Skv, D, generator=g) * case["scale"]
                    v = torch.randn(B, Hkv, Skv, D, generator=g) * case["scale"]
                    for sp in case.get("special", []):
                        _special(q, sp)

                    m = _gen_mask(mask_kind if mask_kind != "null" else None,
                                  B, Hq, (Sq, Skv), dt, g)
                    dm = None
                    if drop_p and drop_p > 0:
                        # 确定性 dropout: 掩码随黄金存档，跨设备可判卷
                        gg = torch.Generator().manual_seed(seed + 5000)
                        keep = torch.rand(B, Hq, Sq, Skv, generator=gg) >= drop_p
                        dm = keep.to(torch.float32) / (1.0 - drop_p)

                    gqa = Hq != Hkv
                    scale = case.get("scale_param")   # 通常 None（1/sqrt(D)）
                    q_t, k_t, v_t = q.to(dt), k_.to(dt), v.to(dt)
                    m_t = _to_dtype(m, dt)

                    expected = ref_fn(q_t, k_t, v_t, m_t, drop_p, causal, dm,
                                      scale=scale, enable_gqa=gqa)

                    # ---------- 三重交叉互验（一律 fp32 语义层） ----------
                    q32, k32, v32 = q.float(), k_.float(), v.float()
                    m32 = m if (m is None or m.dtype == torch.bool) \
                        else m.float()
                    r32 = ref_fn(q32, k32, v32, m32, drop_p, causal, dm,
                                 scale=scale, enable_gqa=gqa)
                    special = bool(case.get("special"))

                    # ① 官方调用路径: F.sdpa(MATH)（不返回 P，只比 out）。
                    #    dropout case F.sdpa 给不出带显式掩码的 P → 由 ②③ 覆盖
                    if not (drop_p and drop_p > 0):
                        from torch.nn.attention import sdpa_kernel
                        with sdpa_kernel(F_sdpa_backends()):
                            x = F.scaled_dot_product_attention(
                                q32, k32, v32, attn_mask=m32,
                                is_causal=causal, enable_gqa=gqa)
                        err = (r32[0] - x).abs()
                        if special:
                            rel = (err / x.abs().clamp_min(1e-3)).nan_to_num(
                                nan=0.0).max().item()
                            assert rel < 1e-3, f"互验①(相对)分歧: rel={rel}"
                        else:
                            e = err.max().item()
                            assert e < 5e-5, f"互验①分歧: abs={e} @seed={seed}"

                    # ② native 直调（本进程未注册 → 原生）: out + P 双比对
                    if m32 is None or m32.dtype != torch.bool:
                        with torch.no_grad():
                            no_, np_ = torch.ops.aten.\
                                _scaled_dot_product_attention_math(
                                    q32, k32, v32, m32, drop_p, causal, dm,
                                    enable_gqa=gqa)
                        if special:
                            rel = ((r32[1] - np_).abs()
                                   / np_.abs().clamp_min(1e-3)).nan_to_num(
                                       nan=0.0).max().item()
                            assert rel < 1e-3, f"互验② P(相对)分歧: {rel}"
                        else:
                            e1 = (r32[0] - no_).abs().max().item()
                            e2 = (r32[1] - np_).abs().max().item()
                            assert e1 < 5e-5 and e2 < 5e-5, \
                                f"互验②分歧: out={e1} P={e2} @seed={seed}"
                    else:
                        # bool mask: native 直调 0/1 加性怪癖（有意分歧，
                        # reference 顶注记录），只做自洽检查
                        pass

                    # ③ 自洽: out == P @ V ; P 行和 (全遮蔽行=0, 其余=1)
                    v_pv = (v32.repeat_interleave(Hq // Hkv, dim=1)
                            if (gqa and Hq != Hkv) else v32)
                    pv = r32[1] @ v_pv
                    if drop_p and drop_p > 0:
                        # 显式 dropout_mask 的 native 怪癖: O = (P/(1-p))@V
                        # （返回的 P 不带 1/(1-p)，O 带）——本实现照抄
                        pv = pv / (1.0 - drop_p)
                    e_pv = (r32[0] - pv).abs().max().item()
                    assert e_pv < 1e-4, f"自洽(O≠P@V 规则)分歧: {e_pv}"
                    if not (drop_p and drop_p > 0):
                        row = r32[1].sum(-1)
                        zero = row.abs() < 1e-6        # 全遮蔽行 → P 必须恒 0
                        if zero.any():
                            assert r32[1][zero].abs().max().item() < 1e-7, \
                                "全遮蔽行 P≠0"
                        keep_row = row[~zero]
                        if keep_row.numel():
                            assert torch.allclose(
                                keep_row, torch.ones_like(keep_row),
                                atol=1e-4), "P 行和≠1"

                    name = (f"{case['name']}_{dt_name}_B{B}_H{Hq}-{Hkv}"
                            f"_S{Sq}x{Skv}_D{D}_c{int(causal)}"
                            f"_m{mask_kind or 'none'}"
                            f"_d{int(drop_p*100)}_s{k}.pt")
                    inputs = [q_t, k_t, v_t]
                    if m_t is not None:
                        inputs.append(m_t)
                    torch.save(
                        {"inputs": inputs,   # q,k,v[,attn_mask]
                         # dropout_mask 单独存（inputs 槽位语义固定，避免
                         # 判卷脚本把它误当 attn_mask）
                         "dropout_mask": dm,
                         "expected": {"out": expected[0].cpu(),
                                      "probs": expected[1].cpu()},
                         "B": B, "Hq": Hq, "Hkv": Hkv, "Sq": Sq, "Skv": Skv,
                         "D": D, "causal": causal,
                         "mask": mask_kind, "dropout_p": drop_p,
                         "dtype": dt_name, "seed": seed, "scale_param": scale,
                         "enable_gqa": gqa, "case": case["name"]},
                        data_dir / name)
                    sha = hashlib.sha256(
                        (data_dir / name).read_bytes()).hexdigest()[:16]
                    index["files"].append({"file": f"data/{name}",
                                           "sha256_16": sha})
                    n += 1
    (Path(args.out) / "index.json").write_text(json.dumps(index, indent=2))
    print(f"生成 {n} 组黄金 -> {data_dir}"
          + (f"（跳过 {skipped_conflict} 组 causal+mask 冲突组合）"
             if skipped_conflict else ""))
    return 0


def F_sdpa_backends():
    """MATH 后端（官方调用路径的强制后端）。"""
    from torch.nn.attention import SDPBackend
    return SDPBackend.MATH


if __name__ == "__main__":
    raise SystemExit(main())
