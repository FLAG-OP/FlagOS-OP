# item（host 标量算子）

`aten::item` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.item` |
| 语义 | 单元素张量取为 Python 标量 (device→host 同步) |
| 判定 | 标量值相等 + Python 类型 + 多元素报错 |

## 运行方式

```bash
cd ops/item
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：aten::item@PrivateUse1（拦截 1 次）。
2. 判定：标量值相等 + Python 类型 + 多元素报错。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

详见 [REPORT.md](REPORT.md)。
