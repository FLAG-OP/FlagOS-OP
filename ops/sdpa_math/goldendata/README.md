# 黄金数据目录

**入库**: 只有 `inputs_spec.yaml`（规格）。
**生成物**（`data/`、`index.json`）体积大，随用随生成，不入库:

```bash
python3 script/gen_golden.py                      # CPU 生成 175 组
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl triton    --device npu:0
python3 script/check_accuracy.py --impl native    --device npu:0   # 原生对照
```

## 规格字段（`inputs_spec.yaml`）

| 字段 | 含义 | 本算子取值 |
|---|---|---|
| `op` | 算子 id | `sdpa_math` |
| `seed_base` | 种子基（每 case 再偏移） | `20260929` |
| `cases[].name/desc` | 用例名与说明 | `basic` / `gqa` / `broadcast` / `nonsq` / `dropout` / `extreme` |
| `B,Hq,Hkv,S,Sq,Skv,D` | 维度（`Sq/Skv` 仅非方阵 case） | 见 [reports/accuracy.md](../reports/accuracy.md) §1 |
| `dtypes` | 精度档 | fp32 / fp16 / bf16 |
| `causal` | `is_causal` 取值集合 | `[false,true]` |
| `mask` | `attn_mask` 形态 | `null` / `bool` / `float` / `bool2d` / `float2d` / `bool_b1` |
| `dropout_p` + 显式 `dropout_mask` | 确定性 dropout（掩码存进黄金） | `0.3`（仅 `dropout` case） |
| `scale` | 显式 `scale` 参数 | `0.5` |
| `seeds_per_case` | 每组合种子数 | 1-2 |
| `special` | 特殊注入 | `zeros_row`（全遮蔽行）、`large_row`（单行饱和） |
| `tolerance.pointwise_abs` | abs 容差分档 | fp32 `1e-5`、fp16/bf16 `2e-2` |
| `tolerance.relative` | `extreme` 用相对容差 | `1e-3` |
| `tolerance.probs_pointwise_abs` | **第二输出概率图**容差 | 同 abs 分档 |

## 生成时的互验（任一不一致即中止）

1. `F.sdpa(MATH)`（`torch.nn.attention.sdpa_kernel` 包裹）的 `out`；
2. 原生直调 `torch.ops.aten._scaled_dot_product_attention_math` 的
   `(out, P)`（bool mask 直调有 0/1 加性怪癖 → 跳过该组数值、仅取输入）；
3. dropout 用例的 native 自洽式 `out == (P/(1-p))@V`；
4. `../reference.py` 参考实现。

`causal + attn_mask` 组合 native 直接报错 → **跳过 32 组**，落盘 175 组。

黄金参考 = `../reference.py` 的 fp32 实现（内部 fp32、输出 cast 回输入
dtype），与 `scripts/accuracy_report.py` 同口径。
