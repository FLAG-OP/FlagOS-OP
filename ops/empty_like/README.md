# empty_like（分配/元数据算子）

`aten::empty_like` 的 MLU（Cambricon）实现，路线 **A1 · torch 算子替换**，
**torch 级**（无 Triton / 硬件级，见 [kernel/README.md](kernel/README.md)）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.empty_like`） |
| 路线 | A1：`Library("aten","IMPL").impl("empty_like", fn, "PrivateUse1")` |
| 语义 | 未初始化张量；只保证 shape/dtype/stride/device，数值不保证 |
| 判定 | **元数据等价**（无数值黄金、无哨兵） |

## 运行方式

```bash
cd ops/empty_like
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 实现用 **meta 设备推 stride** + `empty_strided` 分配，避免调用自身递归。
2. 未初始化 → **不能做数值/哨兵断言**，黄金只比元数据。
3. `memory_format=preserve_format`（默认）时 stride 取自 `self`；显式
   `contiguous_format`/`channels_last` 时按格式推导。

详见 [REPORT.md](REPORT.md)。
