# 精度分册: empty

## 判定口径

`empty` 返回**未初始化**内存，无数值语义，故不做数值判定、无哨兵。
黄金约束**元数据**：(shape, dtype, stride)。参考实现
[reference.py](../reference.py) 为原生 `torch.empty`。

## 黄金数据

规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **48** 组
（4 shape × 3 dtype（contiguous）+ channels_last 组；每组 3 seed）。存参数与参考元数据，不含数值。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
```

## 结果

| 实现 | 通过 | 判定 |
|---|---|---|
| reference（CPU） | 48/48 | 元数据 |
| torch 级（MLU） | 48/48 | shape/dtype/stride 一致 |
