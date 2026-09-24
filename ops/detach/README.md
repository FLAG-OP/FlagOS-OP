# detach（别名/元数据（autograd）算子）

`aten::detach` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.detach` |
| 语义 | 返回与 self 共享底层存储、脱离 autograd 的新张量 (requires_grad=False) |
| 判定 | 共享存储 + requires_grad=False + shape/dtype/stride |

## 运行方式

```bash
cd ops/detach
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：aten::detach@PrivateUse1（拦截 1 次）。
2. 判定：共享存储 + requires_grad=False + shape/dtype/stride。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

详见 [REPORT.md](REPORT.md)。
