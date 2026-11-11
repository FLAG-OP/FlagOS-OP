# matmul 寒武纪 MLU590 平台报告

状态：**兼容性功能与初始性能记录完成；锁定环境复验待做**

寒武纪移植与复验：赵俊堯、张哲、张子煜、赵炳凯。

本文件只记录可以追溯的寒武纪实测证据。未完成项必须保持 PENDING，
不得作为交付通过结论，也不得用昆仑芯 P800 的数字替代。

## 1. 环境

| 项 | 实际值 |
|---|---|
| 日期 | 2026-11-11 |
| 团队 | 赵俊堯、张哲、张子煜、赵炳凯 |
| 服务器 / 容器 | `tza-0a06-ai01-em9` / `mlu-baseline` |
| profile | `mlu590` |
| 设备 | `mlu:0` |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cpu |
| torch_mlu | 1.32.2+torch2.10.0 |
| Triton | 3.4.0+mlu2.0.3 |
| FlagGems | 5.3.5，源码路径 `/root/FlagGems-src/src` |
| commit | 基线 `4df994d` + 本地寒武纪兼容修改 |

说明：当前通用测试容器的 PyTorch、torch_mlu、Triton 和 Python
版本低于 `configs/env/mlu590.lock.yaml` 的锁定值，因此本报告是
兼容性验证证据，不等同于锁定环境的正式 PASS。

## 2. 设备与注册

| 项 | 结果 |
|---|---|
| `torch_mlu` 可导入 | PASS |
| MLU 设备可读回 | PASS，检测到 8 张 MLU |
| `mlu:0` 可用 | PASS |
| A1 dispatch key | `AutogradPrivateUse1` |
| `torch.matmul` 命中 | PASS，15 次 |
| 显式 `aten::matmul` 命中 | PASS |
| CPU 隔离 | PASS |

## 3. 正确性

| 层级 | 结果 |
|---|---|
| 最小矩阵乘前向 | PASS，`63×127 @ 127×65` 误差 `3.17e-4` |
| 最小矩阵乘 dA/dB | PASS，误差分别为 `3.43e-4`、`3.10e-4` |
| 完整 CPU 黄金 | PASS，直调 Triton 与 A1 各 `144/144` |
| 哨兵：确定性 | PASS |
| 哨兵：输入敏感 | PASS |
| 非整 tile shape | PASS |
| 零 K / 空输出 | PASS |
| 非连续输入 | PASS |
| 广播反向 | PASS |
| FP32 / FP16 / BF16 | PASS |
| kernel 最坏误差 | 前向 `6.26e-3`，反向 `7.41e-3` |
| A1 黄金最坏误差 | `7.41e-3`，命中 144 次 |

## 4. 三层测试

| 层 | 命令 | 结果 |
|---|---|---|
| kernel | `python3 ops/matmul/test/kernel_level.py mlu590` | PASS，144 组 |
| op | `python3 ops/matmul/test/op_level.py mlu590` | PASS，命中 15 次，最坏误差 `6.09e-4` |
| framework | `python3 ops/matmul/test/framework_level.py mlu590` | PASS，命中 2 次 |
| 一键三层 | `python3 ops/matmul/example.py mlu590` | PASS |

framework 输出和三个梯度相对误差为
`[3.18e-4, 7.28e-4, 7.04e-4, 2.87e-4]`。

## 5. 性能

| 项 | 实际值 |
|---|---|
| 测试日期 | 待填写 |
| 设备 / 卡号 | 待填写 |
| warmup | 20 |
| iterations | 不超过 100 |
| 进程隔离 | 是 |
| 记录数 | 144/144 成功，0 跳过 |
| 自研前向延迟范围 | `0.235–0.312 ms` |
| 原生前向延迟范围 | `0.040–0.099 ms` |
| 自研反向延迟范围 | `0.477–0.700 ms` |
| 原生反向延迟范围 | `0.171–0.306 ms` |
| 自研/原生几何平均 | 前向约 `5.14–5.31x`，反向约 `2.45–2.49x` |
| 计时前最大精度误差 | `5.0e-3` |
| BF16 A1 分发开销 | native `0.050356 ms`，direct `0.245296 ms`，A1 `0.278312 ms` |
| 基线 | `perf/baselines/mlu590.json` |

完整原始记录见
`reports/cambricon_evidence/results/perf/runs/mlu590/`。基线建立后的
`perf_compare` 是同轮记录的回读结果，不能替代第二次独立性能复跑；
独立复跑和共享机器波动检查仍待完成。

## 6. 问题与限制

1. FlagGems 5.3.5 将 `DeviceDetector` 移至
   `flag_gems.runtime.backend.device_finder`。算子已增加新旧路径兼容导入。
2. `torch_mlu` 的 A1 反向拦截点为 `AutogradPrivateUse1`。注册选择器已
   将 `cambricon` profile 的 `PrivateUse1` 统一提升到该 key。
3. 当前测试容器版本与 `mlu590` 环境锁不一致，正式合并前仍需在锁定栈
   复跑。
4. 当前自研 Triton 在小 shape 上明显慢于 MLU 原生实现。原因是调用和
   Triton/MLU 启动开销突出，尚未做 profiler 分解，本阶段不进行性能优化。

## 7. 结论

寒武纪 MLU 的功能、A1 注册、前向、左右输入梯度、应用消费者、
CPU 黄金回归和 144 条初始性能记录已完成。由于版本未完全匹配环境锁，
且尚无第二次独立性能复跑，暂不将 `cambricon` 登记为正式完整平台交付。
