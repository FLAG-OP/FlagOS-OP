# result_type（dtype 元数据算子）

`aten::result_type` 的 MLU（Cambricon）实现，路线 **自用/实验**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.result_type` |
| 语义 | 两输入在 PyTorch 类型提升下的公共 dtype |
| 判定 | 公共 dtype 相等 |

## 运行方式

```bash
cd ops/result_type
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：—（result_type 为纯函数，不可 A1 注册）。
2. 判定：公共 dtype 相等。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

详见 [REPORT.md](REPORT.md)。
