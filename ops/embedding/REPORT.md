# 算子总体报告: embedding

| 项 | 值 |
|---|---|
| 算子 | `aten::embedding` |
| 范围 | 普通稠密查表 + dense backward |
| 平台 | p800-kunlunxin / torch_xmlir |
| 路线 | A1，`AutogradCUDA` |
| 日期 / 状态 | 2026-09-23 / 三层与黄金全绿 |

## 验证矩阵

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel forward | `test/kernel_level.py` | 19/19，FP32/FP16/BF16 全部 0 error |
| kernel backward | 同上 | 6/6；重复 index、padding、`scale_grad_by_freq` 全部 0 error |
| 黄金 | `script/check_accuracy.py --impl p800` | 117/117，worst=0 |
| A1 op | `test/op_level.py` | 拦截 4 次；hooked=direct 逐位；dense/scale backward 0 error |
| 应用层 | `test/framework_level.py` | `nn.Embedding` + MLP；logits/梯度 0 diff；贪心一致率 1.00 |
| 性能 | `script/bench_perf.py` | 大 shape native parity；Triton 显著慢 |

## 性能摘要

| shape | ours | native | Triton | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0680ms | 0.0358ms | 0.2403ms | 0.526x |
| 16k×D128 | 0.0514ms | 0.0494ms | 2.2206ms | 0.963x |
| 131k×D128 | 0.1665ms | 0.1530ms | 16.9373ms | 0.919x |
| 16k×D512 | 0.0616ms | 0.0548ms | 8.7184ms | 0.889x |
| vocab128k 16k×D128 | 0.0514ms | 0.0515ms | 2.2458ms | 1.001x |

backward 16k×D128：

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 1.3306ms | 1.2993ms | 0.976x |
| scale_grad_by_freq | 1.9562ms | XPU native 不支持 | — |

## 结论

该需求适合“同名接口 + 厂商原生 row-gather 委托”的交付形态。与 SDPA 的
P800 结论一致：backend 存在不等于现有 Triton kernel 竞争力足够。本算子中
native embedding/index-select 已经是最优路径。

## 交付物

| 内容 | 位置 |
|---|---|
| 需求 | [requirement.md](requirement.md) |
| 语义参考 | [reference.py](reference.py) |
| P800 backend | [kernel/p800_kunlunxin.py](kernel/p800_kunlunxin.py) |
| A1 注册 | [register.py](register.py) |
| 一键三层 | [example.py](example.py) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度报告 | [reports/accuracy.md](reports/accuracy.md) |
| 性能报告 | [reports/performance.md](reports/performance.md) |
| 开发报告 | [reports/development.md](reports/development.md) |
