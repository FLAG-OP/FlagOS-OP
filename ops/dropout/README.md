# dropout

`aten::dropout` 的 MLU（Cambricon）实现，路线 **A1**，开发级别 **Triton 级**（Philox RNG）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`F.dropout` / `torch.dropout`） |
| 路线 | A1：`Library("aten","IMPL").impl("dropout", fn, "PrivateUse1")` |
| 设备 | `cambricon`（MLU590-M9） |
| 语义 | train=False/p=0 别名；p=1 全零；其余 `x*mask/(1-p)` |

## 运行方式

```bash
cd ops/dropout
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
2. **RNG 不可位级对齐**: Triton Philox 流与原生不同，随机分支按
   结构+统计判定（值域/比例/seed 可控），确定性分支才逐位比较。
3. RNG 需读并推进 `torch.mlu.default_generators` 状态，使
   `torch.manual_seed` 可控。
4. `F.dropout` 对非法 p 抛 `ValueError`（Python 层），ATen 层抛
   `RuntimeError`——测试 catch 两者。
5. 性能受 RNG 限制，比原生慢 ~5x（与 FlagGems 同级）。

详见 [REPORT.md](REPORT.md)。
