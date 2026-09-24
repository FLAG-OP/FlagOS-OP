# 精度分册: result_type

## 判定口径

公共 dtype 相等。参考实现 [reference.py](../reference.py) 为原生
`torch.result_type` / `self.result_type()`。

## 黄金数据

规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **8** 组。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
```

## 结果

| 实现 | 通过 | 判定 |
|---|---|---|
| reference（CPU） | 8/8 | 公共 dtype 相等 |
| torch 级（MLU） | 8/8 | 一致 |
