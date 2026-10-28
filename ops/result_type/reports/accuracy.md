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

## P800（p800-kunlunxin）复验（2026-10-26）

```bash
python3 script/check_accuracy.py --impl torch --device cuda:1
```

| 实现 | 通过 | 最差 abs err |
|---|---|---|
| torch | 7/7（1 skip） | 0 |

结论：P800 全用例 0 fail；P800 无 fp64，float64 用例按设备能力 skip。
