# empty_strided（分配/元数据算子）

`aten::empty_strided` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级，分配算子无数值/设备码）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.empty_strided` |
| 语义 | 按指定 stride 分配未初始化张量（允许非连续/重叠 stride） |
| 判定 | **元数据等价**（shape/dtype/stride；无数值黄金、无哨兵） |

## 运行方式

```bash
cd ops/empty_strided
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 实现：所需存储 empty + as_strided（避免调用自身递归）。
2. 未初始化 → **不能做数值/哨兵断言**，黄金只比元数据。
3. 注册：aten::empty_strided@PrivateUse1（拦截 2 次）。

详见 [REPORT.md](REPORT.md)。
