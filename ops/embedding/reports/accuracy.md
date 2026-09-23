# embedding 精度报告

## 判卷标准

CPU reference 使用独立 `index_select` 组合，不调用 `F.embedding`。
黄金生成时再与 CPU `F.embedding` 交叉互验，要求 bitwise equal。

## Forward 覆盖

| 维度 | 值 |
|---|---|
| dtype | FP32 / FP16 / BF16 |
| index rank | 1D / 2D / 3D / empty |
| embedding dim | 8 / 80 / 512 |
| padding | none / valid padding_idx |
| index dtype | int64 / int32 |
| duplicates | 显式重复 index |

结果：

```text
kernel 20/20 max_err=0
golden 117/117 max_err=0
```

## Backward 覆盖

- FP32 / FP16 / BF16；
- repeated indices；
- padding row；
- `scale_grad_by_freq=False/True`。

结果：

```text
kernel backward 6/6 max_err=0
A1 dense grad max_err=0
A1 scale_grad_by_freq grad max_err=0
application embedding grad max_err=0
```

## 复现

```bash
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 test/kernel_level.py p800-kunlunxin
python3 test/op_level.py p800-kunlunxin
```
