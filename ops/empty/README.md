# empty（分配/元数据算子）

`aten::empty` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级，分配算子无数值/设备码）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.empty` |
| 语义 | 分配 size 形状的未初始化张量，只保证元数据 |
| 判定 | **元数据等价**（shape/dtype/stride；无数值黄金、无哨兵） |

## 运行方式

```bash
cd ops/empty
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 实现：meta 推 stride + empty_strided（避免调用自身递归）。
2. 未初始化 → **不能做数值/哨兵断言**，黄金只比元数据。
3. 注册：—。

详见 [REPORT.md](REPORT.md)。
