# contiguous

`aten::contiguous` 的 MLU（Cambricon）实现，路线 **A1**，开发级别 **Triton 级**。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.Tensor.contiguous`） |
| 路线 | A1：`Library("aten","IMPL").impl("contiguous", fn, "PrivateUse1")` |
| 设备 | `cambricon`（MLU590-M9） |
| 语义 | 已连续返回 self（别名）；否则新连续张量；仅覆盖 contiguous_format |

## 运行方式

```bash
cd ops/contiguous
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
2. 转置型 2D 必须走分块 swiz（`_contig_2d_swiz`），否则逐元素 gather
   慢 ~80x（见 reports/performance.md）。
3. channels_last 等 memory_format 走原生回退。
4. 参考实现用原生 `.contiguous`：op 层需在注册前取原生参考。

详见 [REPORT.md](REPORT.md)。
