# 性能报告: matmul

## 1. 配置

| 项 | 值 |
|---|---|
| 日期 / 复核 | 2026-10-14～2026-10-15 / 2026-10-22 |
| 环境 | 个人容器 `tyanfeng-flagos-p800-lock`；Python 3.10.18、Torch 2.9.0+cu129、Triton 3.0.0+03c4c9be |
| 设备 | P800 物理 XPU 2；`CUDA_VISIBLE_DEVICES=1,2`，逻辑 `cuda:1` |
| 算法 | 自研 Triton dot，动态 K；`XMLIR_MATMUL_FAST_MODE=0` |
| 精度 / 方向 | FP32、FP16、BF16；前向、双输入一阶反向 |
| 计时 | 每条用例独立进程；20 次预热、100 次调用取平均；每次读回结果首元素 |
| 入口 | `example.py::perf_cases` → `script/perf_common.py`；`script/bench_perf.py` 复用相同输入与计算定义 |

计时包含 Python 调用、输出分配、设备计算及 `.item()` 读回；不是纯设备
kernel 时间。输入生成、传输、精度检查和 JIT 预热不进入计时。
共享服务器其他卡仍有任务，未锁频；记录的是当前共享环境下的调用延迟。

## 2. 测试数据与覆盖

输入来自 [inputs_spec.yaml](../goldendata/inputs_spec.yaml)，使用黄金生成器
`make_case`，保留布局、特殊值和缩放。每个形状/dtype 取该黄金组合的
第一个固定 CPU 种子，各实现重新生成相同 A、B、上游梯度。普通输入为
正态随机数乘 0.25，上游梯度为正态随机数；`large_input` 输入乘 16，
`zero_input` 的 A 置零。全部 36 组输入定义与种子见
[workloads.json](performance_xpu2_v1/workloads.json)。

| 用例 | A shape | B shape | 覆盖 |
|---|---|---|---|
| matrix_tail | (63,127) | (127,65) | 非整 tile |
| matrix_regular | (128,256) | (256,192) | 规则矩阵 |
| batch_tail | (2,33,65) | (2,65,47) | 批量矩阵 |
| batch_broadcast | (2,1,33,65) | (1,3,65,47) | 批次广播与反向归约 |
| dot | (67,) | (67,) | 向量点积 |
| matrix_vector | (37,65) | (65,) | 矩阵乘向量 |
| vector_matrix | (65,) | (65,29) | 向量乘矩阵 |
| noncontiguous | (65,37) | (37,29) | A 为转置视图 |
| zero_input | (7,17) | (17,9) | 零值输入 |
| large_input | (16,127) | (127,11) | 大值输入 |
| matrix_tiny | (2,2) | (2,2) | 微小矩阵 |
| rank_one | (37,1) | (1,65) | K=1 |

共 **12 形状 × 3 dtype × 2 方向 × 2 实现 = 144 条性能记录/轮**。
零 K、空 M、空 N、空 batch 四类不执行有效 GEMM，保留精度测试，未纳入
本次 GEMM 性能基线。性能输入为合成数据，未使用真实模型。

精度测试的 **144 组**是 16 类 × 3 dtype × 3 种子，每组检查前向、dA、dB；
与这里的 144 条性能记录不是同一个计数。此前性能入口只有两个 BF16 前向
形状，不能代表已覆盖完整精度用例，本次已扩展。

### 前向与反向的对比口径

- 自研前向：`matmul_triton`；原生前向：当前 PyTorch/XMLIR 后端的
  `torch.matmul`，输入不要求梯度。
- 自研反向：`matmul_backward_triton`，计算 dA+dB 及广播归约；原生反向：
  `torch.autograd.grad`，返回同样两个梯度。原生前向图在计时前构建并复用，
  不累计 `.grad`。原生含 autograd 引擎开销，自研直调不含 A1 包装；
  本表比较两种反向入口的调用延迟。
- 每次反向分别读回两个梯度首元素，确保两者完成。计时前对 CPU FP32
  内部参考检查形状、有限值及误差；非零参考最大相对误差 ≤ 5e-2，零参考
  采用 FP32 1e-5、FP16/BF16 1e-2 绝对容差。
- GFLOPS 按前向 `2BMKN`、反向 `4BMKN` 计算，未计广播归约额外 FLOPs。

## 3. 性能结果与 A1 开销

本次完整基线轮 **144/144 条成功**，每条计时前的前向或双梯度精度检查均通过。
以下是基线轮的范围与跨形状汇总，完整 36 行数据见 [逐项对照表](performance_xpu2_v1/results.md)。

| dtype | 方向 | 自研延迟范围 ms | 原生延迟范围 ms | 自研/原生耗时几何平均 |
|---|---|---:|---:|---:|
| float32 | 前向 | 0.091385–0.152901 | 0.036316–0.085793 | 2.16x |
| float32 | 反向 dA+dB | 0.178073–0.326345 | 0.150817–0.334902 | 0.87x |
| float16 | 前向 | 0.093202–0.145185 | 0.036585–0.083591 | 2.11x |
| float16 | 反向 dA+dB | 0.182307–0.341237 | 0.146596–0.420011 | 0.97x |
| bfloat16 | 前向 | 0.094722–0.145493 | 0.035544–0.117754 | 1.88x |
| bfloat16 | 反向 dA+dB | 0.179740–0.323051 | 0.155564–0.579039 | 0.80x |

比值大于 1 表示自研入口更慢。本轮 36 组前向均慢于原生；反向有 23/36
组自研入口耗时更短，但原生含 autograd 引擎成本，不能据此宣称自研
GEMM kernel 更快。后续复测已显示部分原生耗时有明显变化，这些单轮比值
尚不能证明稳定性能优势。

本组形状普遍较小，自研前向约 0.1 ms 的调用成本较突出，可能涉及
Python/设备启动及读回开销；尚未用 profiler 分离成本，不能精确归因。
这属于后续性能周期的优化候选，本阶段只保留现状和测量口径。保留动态 K 正确性修复；
此次没有为速度切换 fast mode 或降低精度。结构化记录最大检查误差约
0.007，低于相对阈值 0.05；该字段按仓库格式保留三位小数，精确正确性
结论仍见 [精度报告](accuracy.md)。

XPU 2 A1 分发开销已补测。测量采用二维 `(63,127) @ (127,65)`、CPU
种子 20260930、输入需要梯度，各路径独立进程、20 次预热和 100 次采样；
它与上述不要求梯度的前向表口径不同。A1 减去直调的差包含注册 wrapper
与 autograd 包装，不是纯 dispatcher 时间。

| dtype | 原生中位数 ms | 直调 Triton ms | A1 入口 ms | A1 - 直调 ms |
|---|---:|---:|---:|---:|
| float32 | 0.044249 | 0.098758 | 0.128068 | 0.029310 |
| float16 | 0.047634 | 0.099998 | 0.130267 | 0.030269 |
| bfloat16 | 0.053644 | 0.098593 | 0.129502 | 0.030909 |

分 dtype 的分发数据见 `performance_xpu2_v1/dispatch_xpu2_*_63x127x65.json`。
同一新版性能入口的广播反向代表用例也已验证：BF16
`batch_broadcast` 的 Triton/native 分别为 0.3145/0.3247 ms，FlagGems
反向因脚本明确标记为未验证路径而记录为 unavailable。


锁定 FlagGems 4.2.1rc0 仅暴露 `mm`、`bmm`，没有完整 `linalg_matmul`
入口。此前 `bmm` 自动调参在 `triton.testing.do_bench` 报
`ZeroDivisionError`，未得到有效性能；本次 144 条基线只纳入自研与原生，
不把 FlagGems 的缺失值算为通过。FlagGems 反向未作为已验证性能对照。

## 4. 基线与回归门禁

基线文件：[p800-kunlunxin.json](../../../perf/baselines/p800-kunlunxin.json)。
每个 case 有自己的历史实测耗时；自研和原生分别登记。建立基线用于检查
未来代码是否退化，不表示自研已经超过原生。

本次只合并新增 `ops.matmul.*`，保留其他算子已有记录。
完成一轮 144 条且零跳过后保存基线，再独立重跑 144 条进行比较。
使用仓库默认阈值：相对基线变慢超过 20% 为 WARN，超过 30% 为 FAIL。

**当前结论：基线已建立，完整独立复测已完成；门禁结果为 FAIL 4 / WARN 4，
不能标记通过。**

独立复测最终完成 **144/144** 条。
完整门禁按仓库默认阈值计算为：**OK 136 / WARN 4 / FAIL 4**。
其中自研 Triton 72 条全部 OK；8 条 WARN/FAIL 全部来自原生路径。
完整结果见 [gate.md](performance_xpu2_v1/gate.md)。

| 超阈值用例（均 native） | 基线 ms | 复测 ms | 变化 | 状态 |
|---|---:|---:|---:|---|
| matrix_regular.float16.forward | 0.041121 | 0.056796 | +38.1% | FAIL |
| matrix_vector.float32.backward | 0.150817 | 0.221842 | +47.1% | FAIL |
| vector_matrix.float16.backward | 0.204462 | 0.251251 | +22.9% | WARN |
| vector_matrix.bfloat16.backward | 0.167057 | 0.210679 | +26.1% | WARN |
| noncontiguous.float16.backward | 0.165623 | 0.234561 | +41.6% | FAIL |
| noncontiguous.bfloat16.backward | 0.195719 | 0.266999 | +36.4% | FAIL |
| zero_input.bfloat16.backward | 0.178073 | 0.228168 | +28.1% | WARN |
| large_input.float32.backward | 0.193877 | 0.246784 | +27.3% | WARN |

上述 8 条原生异常各追加 3 次独立测量。4 条重复均值回到基线 ±10% 内，
另外 4 条均值仍约高于基线 23%–30%，但单次结果波动明显，3 次样本不足以
建立稳定回归结论。复核表和原始数据见
[anomaly_repeats.md](performance_xpu2_v1/anomaly_repeats.md)。因此保留
本次完整门禁的 FAIL/WARN 判定，不修改基线、不放宽阈值，也不把原生异常
归因给自研 Triton。


本次提交的正式性能记录、输入定义、源码哈希和门禁摘要归档在
[performance_xpu2_v1/](performance_xpu2_v1/)。

本报告只采用正式物理 XPU 2 的性能记录；K=1 修复专项记录与当前基线分开，
用于解释正确性修复，不作为性能门禁数据。

## 5. 复现

在个人容器项目根目录执行，先确认物理 XPU 2 可用：

```bash
export CUDA_VISIBLE_DEVICES=1,2
export XMLIR_MATMUL_FAST_MODE=0
python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul.
python scripts/perf_compare.py --device p800-kunlunxin
python ops/matmul/script/bench_perf.py --device p800-kunlunxin --case batch_broadcast --dtype bfloat16 --direction backward
python ops/matmul/script/bench_dispatch.py --device p800-kunlunxin --dtype bfloat16
```

日常回归不要加 `--update-baseline`；只有明确接受新的实现或环境基准，且
确认全部目标用例成功时才更新。其他国产平台尚未实测，不能沿用 P800 基线。

## 6. 结论与遗留

| 项目 | 结论 |
|---|---|
| 功能性能入口 | 144 条基线和 144 条独立复测均完成，计时前精度检查全部通过 |
| 自研 Triton | 72 条门禁记录均 OK；本阶段不进行性能优化 |
| 原生 PyTorch/XMLIR | 4 FAIL / 4 WARN；追加复核显示共享设备上的延迟波动，不能直接归因于 matmul 代码 |
| FlagGems | 锁定版本缺少完整 matmul 入口，`bmm` 自动调参失败；原因已记录，不构造缺失数字 |
| A1 分发 | 三种 dtype 已完成独立进程测量，结果见本报告第 3 节 |
| 其他平台 | 寒武纪、昇腾、壁仞没有沿用本基线，需各自环境独立采集 |

性能数据用于回归追踪，不代表自研实现已经超过原生；当前第一阶段的功能和
正确性结论见 [测试报告](test-report.md) 与 [精度报告](accuracy.md)。
