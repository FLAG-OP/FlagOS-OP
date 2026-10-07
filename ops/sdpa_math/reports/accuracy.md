# 精度报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> 数据来源: `python3 script/check_accuracy.py --impl <名> --device <profile>`
> （读 [goldendata](../goldendata/README.md)）。黄金在 **CPU fp32** 生成
> （`script/gen_golden.py`），跨设备复用同一份 175 组。

## 1. 用例覆盖

规格: [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)。
每 case 按 `B × Hq/Hkv × S × D × dtype × causal × mask × seeds` 展开，
`causal + mask` 组合按 native 规则**跳过 32 组**（冲突报错），落盘 175 组。

| case | shape(s) | dtype | 分布 / special | 组数 |
|---|---|---|---|---|
| basic | B1/B2 H4/4 S128·100 D64 | fp32/fp16/bf16 | randn×0.5；causal×{null,bool,float}；2 seed | 96 |
| gqa | B1 H8/2 S128·197 D64·80 | fp32/fp16 | GQA 广播 + ViT 长序列；2 seed | 48 |
| broadcast | B2 H4/4 S64 D64 | fp32/fp16 | mask 形态 bool2d / float2d / bool(B,1,S,S) | 6 |
| nonsq | B1 H4/4 Sq1·6 × Skv64·4 D64 | fp32/fp16 | 非方阵（causal 左上 tril / decode） | 16 |
| dropout | B1 H4/4 S64 D64 | fp32/fp16 | **显式 dropout_mask 存档**（跨设备确定），p=0.3 | 8 |
| extreme | B1 H4/4 S128 D64 | fp32 | `zeros_row` + `large_row`（单行饱和） | 1 |
| **合计** | | | | **175**（+32 组 causal×mask 跳过） |

按 dtype：fp32 72 · fp16 71 · bf16 32。容差
（`tolerance.pointwise_abs` / `probs_pointwise_abs`）：fp32 `1e-5`、
fp16/bf16 `2e-2`（abs）；extreme 用 `relative 1e-3`。

**生成期四重互验**（任一不一致即中止）：
① `F.sdpa(MATH)`（`sdpa_kernel` 包裹） ② 原生直调的 `(out, P)`
③ dropout 用例的自洽式 `out == (P/(1-p))@V` ④ 参考实现。

## 2. 结果

| 实现 | 设备 | 结果 | 最差 abs（out） | 最差 abs（P） | 判定 |
|---|---|---|---:|---:|---|
| 自研 `triton` | npu:0 | **175/175** | 3.906e-3（bf16） | 1.118e-8~ | ✓ |
| 自研 `torch` | cpu | **175/175** | 1.953e-3（bf16） | 3.7e-9~ | ✓ |
| P800 `p800` | cuda:1 | **175/175** | 3.906e-3（bf16） | 3.906e-3（bf16） | ✓ |
| 参考 `reference` | cpu | **175/175** | 0 | 0 | ✓ |
| 原生 `native` | npu:0 | **131/131**（44 跳过） | 9.766e-4（bf16） | 同档 | ✓ |
| FlagGems | — | 不适用 | — | — | 无"返回概率图"的同语义实现 |

最差值均出现在 bf16（量化噪声本身 ~1e-2），远低于 2e-2 容差；fp32 全部
≤1.2e-7 量级。

`--impl native` 的 44 组跳过 = 本算子 **唯一有意分歧**（见 §3），
逐 case 打印 `SKIP(bool 直调怪癖)`。

## 3. 关键精度决策（均有实测依据）

| 决策 | 依据 |
|---|---|
| `bool attn_mask` 按 `-inf` 遮蔽，不抄 native 直调的 0/1 加性 | `probes/native_semantics.py` §3：直调 vs 0/1 加性 `0.00e+00`，vs `-inf` 遮蔽 `5.14e-01`；而 `F.sdpa` 到达本算子前已把 bool 转成 float32 `-inf` 加性（§5 记录：`range=[-inf,0.0]`）→ 真实消费方只可能见到遮蔽语义 |
| 内部 fp32，fp64 输入保持 fp64 | native 同口径；fp64 是 gradcheck 的唯一可行精度（`register.py` backward `acc` 选择） |
| P800 fast 只在 fp16/bf16 无 mask/dropout 时启用 | `efficient_attention` 产出 O/LSE，`P=exp(scale·QKᵀ-LSE)`；fp32、mask、dropout 与 direct-autograd 回退 `torch_level`。黄金 175/175，bf16 worst 3.9e-3 < 2e-2 |
| 极端 case 从 `scale=8.0` 改 `0.5` | `scale=8` 令全矩阵饱和、任意两行并列时 fp32 累加顺序差一点就翻转 argmax，跨实现相对误差 2.3e-2——病态输入不可判卷（`inputs_spec.yaml` 注释）；改后 single-row 饱和仍在（`large_row=40`），相对容差 1e-3 可用 |
| 全遮蔽行 → `P=0、O=0`（不是 NaN） | `probes` §4：float mask 整行 `-inf` → `max\|P\|=0.00e+00`、输出 finite；torch_level 用 `nan_to_num(softmax)` 落地 |
| `causal + attn_mask` 报错（bool/float 皆然） | `probes` §4 两种 dtype 均抛 `Explicit attn_mask should not be set...` |
| 黄金同时判 `out` 与 `attn_probs` | 第二输出是本算子的核心契约（`F.sdpa` 丢弃它），只判 out 会漏掉 dP 消费方的错误 |

## 4. kernel 层直测（52 组/profile，与黄金互补）

16 组 shape（含 S=100/197 非整倍数、D=80 非 2 幂、Sq1×Skv7、Sq6×Skv4、
Sq4×Skv6、GQA 8/2、`Hkv=1`）× 3 dtype + 掩码四形态 + 全遮蔽 + 错误路径。

| dtype | NPU max\|Δout\| | CPU max\|Δout\| | max\|ΔP\| | 容差 | 判定 |
|---|---:|---:|---:|---:|---|
| fp32 | 1.79e-7 | 2.38e-7 | 1.22e-4 | 1e-5 | ✓ |
| fp16 | 1.95e-3 | 1.22e-4 | 1.22e-4 | 2e-2 | ✓ |
| bf16 | 7.81e-3 | 4.88e-4 | 1.22e-4 | 2e-2 | ✓ |

dropout 专项：显式掩码与参考逐位一致（fp32 1e-5 内）且满足
`O=(P/(1-p))@V`；随机路径 keep 率 0.499、kept 上 ×2.000。

## 5. 失败分析（本次开发中的真实踩坑）

| case | 现象 | 定位过程 | 根因 | 关联 known-issue |
|---|---|---|---|---|
| dropout（初版黄金） | `reference/torch/triton` 三方差 ~2e-1 | 用显式掩码五种幅值（0/1/-1/5/-3）扫描 `P`、`O` 相对 `P0` 的因子表 | 抄成单一规则：native 显式路径 `P` **不缩放**、`O` 缩放 `1/(1-p)`，与随机路径不同 | —（native 语义，固化进 reference 顶注） |
| extreme（scale=8.0） | 跨实现相对误差 2.3e-2 | 逐行比对 argmax 翻转位置 | 全矩阵饱和 → 近并列行被累加顺序左右 | 判卷输入病态（改 `scale=0.5`） |
| bool mask 直调对照 | 与 native 差 9.85e-1 | 分别用 0/1 加性与 `-inf` 遮蔽重算 | native **直调**按 0/1 加性；`F.sdpa` 到达时已是 `-inf` | —（有意分歧，见 §3） |
| 掉注册 | 拦截 counter 恒为 0 | 保留 `register_a1` 返回值后恢复 | `torch.library` 对象被 GC → 注册回收 | —（register.py docstring 已标注，测试均持有引用） |

## 6. 复现

```bash
python3 script/gen_golden.py                                  # 175 组（CPU，四重互验）
python3 script/check_accuracy.py --impl reference --device cpu   # 175/175
python3 script/check_accuracy.py --impl torch     --device cpu   # 175/175
python3 script/check_accuracy.py --impl triton    --device npu:0 # 175/175
python3 script/check_accuracy.py --impl p800      --device cuda:1 # 175/175
python3 script/check_accuracy.py --impl native    --device npu:0 # 131/131 + 44 bool 跳过
python3 test/kernel_level.py ascend910                         # 52 组直测（精度表）
python3 probes/native_semantics.py ascend910                   # native 语义证据 6 节
```
