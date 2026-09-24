# _local_scalar_dense（host 标量算子）

`aten::_local_scalar_dense` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch._local_scalar_dense` |
| 语义 | item 的底层原语：单元素稠密张量 → host 标量 |
| 判定 | 标量值相等 + Python 类型 + 多元素报错 |

## 运行方式

```bash
cd ops/_local_scalar_dense
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：aten::_local_scalar_dense@PrivateUse1（拦截 1 次）。
2. 判定：标量值相等 + Python 类型 + 多元素报错。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

详见 [REPORT.md](REPORT.md)。
