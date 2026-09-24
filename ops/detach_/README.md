# detach_（别名/元数据（autograd）算子）

`aten::detach_` 的 MLU（Cambricon）实现，路线 **自用/实验**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.detach_` |
| 语义 | 原地脱离 autograd，返回 self (requires_grad=False) |
| 判定 | 原地返回 self + requires_grad=False + 存储不变 |

## 运行方式

```bash
cd ops/detach_
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：—（detach_ 非 backend dispatch，不可 A1 注册）。
2. 判定：原地返回 self + requires_grad=False + 存储不变。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

详见 [REPORT.md](REPORT.md)。
