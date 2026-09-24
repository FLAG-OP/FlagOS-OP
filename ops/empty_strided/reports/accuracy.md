# 精度分册: empty_strided

## 判定口径

`empty_strided` 返回**未初始化**内存，无数值语义，故不做数值判定、无哨兵。
黄金约束**元数据**：(shape, dtype, stride)。参考实现
[reference.py](../reference.py) 为原生 `torch.empty_strided`。

## 黄金数据

规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **30** 组
（6 组 shape/stride（含 3D、重叠 stride (1,0)、零元素）× dtype；每组 3 seed）。存参数与参考元数据，不含数值。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
```

## 结果

| 实现 | 通过 | 判定 |
|---|---|---|
| reference（CPU） | 30/30 | 元数据 |
| torch 级（MLU） | 30/30 | shape/dtype/stride 一致 |
