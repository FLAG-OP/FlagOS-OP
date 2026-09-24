# copy_

`aten::copy_` 的 MLU（Cambricon）实现，路线 **A1**，开发级别 **Triton 级**。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.Tensor.copy_`） |
| 路线 | A1：`Library("aten","IMPL").impl("copy_", fn, "PrivateUse1")` |
| 设备 | `cambricon`（MLU590-M9） |
| 语义 | 原地写 dst 并返回 dst；src 支持广播 + cast 到 dst.dtype |

## 运行方式

```bash
cd ops/copy_
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
2. **原地语义**: 必须返回 dst 本身；测试每次新建 dst。
3. **全局注册侵入性强**: copy_ 是高频内部算子，框架层用子进程隔离；
   op 层需在注册前取原生参考。
4. boundary 溢出到 ±inf 属合法 cast，判定用 isclose。
5. 自拷贝（别名）走原生回退；2D 转置型走分块 swiz。

详见 [REPORT.md](REPORT.md)。
