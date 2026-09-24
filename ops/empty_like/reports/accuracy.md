# 精度分册: empty_like

## 判定口径

`empty_like` 返回**未初始化**内存，无数值语义，故不做数值判定、无哨兵。
黄金约束**元数据**：(shape, dtype, stride)。参考实现
[reference.py](../reference.py) 为原生 `torch.empty_like`。

## 黄金数据

规格 [goldendata/inputs_spec.yaml](../goldendata/inputs_spec.yaml)，共 **168** 组
（4 shape × 3 dtype × {preserve, contiguous} × {contig, transpose} × 3 seed，
另 channels_last 组）。存 self 与参考元数据，不含数值。

```bash
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl reference --device cpu
python3 script/check_accuracy.py --impl torch     --device mlu
```

## 结果

| 实现 | 通过 | 判定 |
|---|---|---|
| reference（CPU） | 168/168 | 元数据 |
| torch 级（MLU） | 168/168 | shape/dtype/stride 一致 |

## 结构性检查（kernel 层）

- preserve_format: stride == self.stride()（contig 与 transpose）。
- contiguous_format: 输出连续；channels_last: 与原生一致。
- 存储独立: `out.data_ptr() != self.data_ptr()`。
- 零元素 shape 保持。
