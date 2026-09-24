# type_as（算子库层 / 框架层 / 应用层）

`aten::type_as` 的 MLU（Cambricon）实现，路线 **A1 · torch 算子替换**，
开发级别 **Triton 级**。骨架从 [templates/operator](../../templates/operator/)
复制并按算子改造。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.Tensor.type_as`） |
| 路线 | A1：`torch.library.Library("aten","IMPL").impl("type_as", fn, "PrivateUse1")` |
| 设备 | `cambricon`（MLU590-M9，PrivateUse1=mlu），profile 见 [configs/devices/cambricon.yaml](../../configs/devices/cambricon.yaml) |
| 语义 | `out = self.to(other.dtype)`，留在 `self.device`；同 dtype 返回 `self` |

## 运行方式

```bash
cd ops/type_as
python3 example.py cambricon                    # 一键三层

python3 test/kernel_level.py cambricon          # 算子库层：精度/哨兵/性能
python3 test/op_level.py cambricon              # 框架层：aten 注册+拦截
python3 test/framework_level.py cambricon       # 应用层（子进程）

python3 script/gen_golden.py --device cpu       # 黄金（CPU 参考）
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```

## 预期输出

```
1/3 算子库层 PASS  {'ok': True, 'max_err': 0.0, ...}
2/3 框架层   PASS  {'ok': True, 'registered': 'aten::type_as@PrivateUse1', ...}
3/3 应用层   PASS  {'ok': True, 'calls': 1, 'output_match': True, ...}
```

## 关键点 / 坑

1. **#11**: Triton 启动包 `torch_device_fn.device(self.device)`，否则首次后静默 no-op。
2. 无 `tl.sum`，**#15a 尾块归约坑不适用**；不用 `@triton.autotune`（#15b）。
3. 非 dense 的 fp16/bf16→fp32（窄→宽）有 MLU codegen bug，走 `out.copy_(self)`
   兜底；dense/affine 路径不受影响。
4. MLU 栈要求**先 `import torch` 再导入 triton**（否则 torch.library 的
   triton 命名空间重复注册）；已在 `common/device.py` 处理。
5. 本机未装 vLLM，应用层用等价 torch API 的**子进程**验证（注册不跨进程
   传播这一核心难点仍被覆盖），vLLM 到位后替换 `_app_child` 即可。

详见 [REPORT.md](REPORT.md) 与 [reports/](reports/)。
