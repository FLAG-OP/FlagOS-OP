# 平台绑定清单（Platform Binding）— matmul

> 当前只保留有运行证据的平台结论。昆仑芯 P800 已有功能与精度证据；
> 寒武纪 MLU590 已完成兼容性环境的功能、精度和初始性能复验，锁定
> 环境复验仍待完成。

## 1. 平台无关复用项

| 内容 | 位置 |
|---|---|
| FP32 语义参考 | `reference.py` |
| 向量升降维、批次广播和一阶反向 | `kernel/_shape.py` |
| CPU 黄金规格与生成器 | `goldendata/`、`script/gen_golden.py` |
| kernel/op/framework 测试 | `test/` |
| 精度和性能脚本 | `script/` |

## 2. 昆仑芯 P800

已完成范围以 [REPORT.md](REPORT.md) 为准：

- A1 dispatch key：`AutogradCUDA`；
- 自研 Triton dot；
- 前向和双输入一阶反向；
- 144/144 精度组合；
- 144 条性能基线；
- 性能门禁仍为 PARTIAL，不能跨平台复制。

昆仑芯 BF16 `K=1` 数据残留问题由
`@triton.jit(do_not_specialize=["K"])` 修复。该结论只表示 P800 行为，
其他芯片仍需单独检查。

## 3. 寒武纪 MLU590

状态：**兼容性 PASS；锁定环境复验待做**

| 项 | 目标值 / 复验结果 |
|---|---|
| profile | `mlu590` |
| 设备 | `mlu:0` |
| runtime | `torch_mlu` |
| A1 dispatch key | `AutogradPrivateUse1` |
| kernel | 复用自研 `tl.dot` 主体 |
| 前向 | PASS |
| 左输入梯度 | PASS |
| 右输入梯度 | PASS |
| FP32 / FP16 / BF16 | PASS |
| 零 stride 梯度 | PASS |
| 小型 framework 消费者 | PASS，命中 2 次 |
| 性能基线 | 144 条初始记录；自研前向约慢 5.1–5.3x，反向约慢 2.45–2.49x |

移植时重点检查：

1. `torch_device_fn.device(a.device)` 是否能包住 MLU 上的 kernel 启动；
2. `tl.dot` 的 dtype 组合和 tile 配置是否合法；
3. 仅注册 `PrivateUse1` 是否漏掉反向；
4. `torch.matmul`、显式 `aten::matmul` 和 CPU 隔离是否都正确；
5. `mlu:0` 是否为当前可见设备，且没有其他任务占用。

实际复验使用共享容器 `mlu-baseline`：PyTorch 2.10.0、
torch_mlu 1.32.2、Triton 3.4.0+mlu2.0.3、FlagGems 5.3.5 源码。
该版本低于 `mlu590` 环境锁，不能把兼容性结果写成锁定环境正式通过。

## 4. 新平台准入

按 [MERGE.md](MERGE.md) 执行。必须同时提供：

- 独立环境记录；
- CPU 黄金回归；
- A1 实际命中；
- kernel/op/framework 三层证据；
- 独立性能采集；
- 平台限制和已知问题。

