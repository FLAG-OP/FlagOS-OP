# 精度报告: embedding

## 1. 用例覆盖

判卷标准：

- CPU reference 使用独立 `index_select` 组合，不调用 `F.embedding`；
- 黄金生成时与 CPU `F.embedding` 交叉互验；
- 要求 forward 与 golden 均 bitwise equal。

Forward 覆盖：

| 维度 | 值 |
|---|---|
| dtype | FP32 / FP16 / BF16 |
| index rank | 1D / 2D / 3D / empty |
| embedding dim | 8 / 80 / 512 |
| padding | none / valid padding_idx |
| index dtype | int64 / int32 |
| duplicates | 显式重复 index |
| sparse 参数 | `sparse=True` forward 与 dense forward bitwise equal |

Backward 覆盖：

| 维度 | 值 |
|---|---|
| dtype | FP32 / FP16 / BF16 |
| indices | repeated indices |
| padding | valid `padding_idx` |
| scale | `scale_grad_by_freq=False/True` |
| sparse | backward 显式拒绝 |

## 2. 结果

```text
kernel forward 20/20 max_err=0
kernel backward 6/6 max_err=0
golden 174/174 max_err=0
A1 dense grad max_err=0
A1 scale_grad_by_freq grad max_err=0
application embedding grad max_err=0
```

复现：

```bash
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 script/check_accuracy.py --impl native --device cuda:1
```

## 3. 失败分析（如有）

无失败用例。
