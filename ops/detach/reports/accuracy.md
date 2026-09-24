# 精度分册: detach

## 判定口径

共享存储 + requires_grad=False + shape/dtype/stride。参考实现 [reference.py](../reference.py) 为原生
`torch.detach` / `self.detach()`。

## 黄金数据

规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **36** 组。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
```

## 结果

| 实现 | 通过 | 判定 |
|---|---|---|
| reference（CPU） | 36/36 | 共享存储 + requires_grad=False + shape/dtype/stride |
| torch 级（MLU） | 36/36 | 一致 |
