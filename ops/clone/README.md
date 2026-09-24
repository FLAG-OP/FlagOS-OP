# clone

`aten::clone` 的 MLU（Cambricon）实现，路线 **A1**，开发级别 **Triton 级**。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.Tensor.clone`） |
| 路线 | A1：`Library("aten","IMPL").impl("clone", fn, "PrivateUse1")` |
| 设备 | `cambricon`（MLU590-M9） |
| 语义 | 独立新张量，shape/dtype/device 同 self，不共享存储；preserve_format |

## 运行方式

```bash
cd ops/clone
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```

## 关键点 / 坑

1. **#11**: Triton 启动包 `torch_device_fn.device`。
2. 无归约，不涉 #15a；不用 autotune（#15b）。
3. 参考实现用原生 `.clone`：op 层测试需在**注册前**取原生参考。
4. 未实现硬件级（无厂商 clone 原语）。

详见 [REPORT.md](REPORT.md)。
