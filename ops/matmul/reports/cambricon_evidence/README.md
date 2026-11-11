# 寒武纪 MLU590 原始证据

采集日期：2026-11-11。

寒武纪移植与复验：赵俊堯、张哲、张子煜、赵炳凯。

| 路径 | 内容 |
|---|---|
| `results/cambricon/gen_golden.log` | 144 组 CPU 黄金生成 |
| `results/cambricon/kernel_level.log` | kernel 精度、哨兵和零 stride 测试 |
| `results/cambricon/accuracy_triton.log` | 直调 Triton 黄金回归 |
| `results/cambricon/accuracy_a1.log` | A1 黄金回归和命中计数 |
| `results/cambricon/framework_level.log` | 小型应用消费者 |
| `results/cambricon/example.log` | kernel / op / framework 一键复验 |
| `results/cambricon/perf_update.log` | 144 条性能记录 |
| `results/cambricon/perf_compare.log` | 初始基线对比 |
| `results/cambricon/dispatch_bf16.log` | A1 分发开销 |
| `results/perf/runs/mlu590/` | 每条性能用例的原始 JSON |
| `perf/baselines/mlu590.json` | 寒武纪初始基线 |
| `env_snapshot.md` | 环境快照 |

本证据来自无个人挂载的共享容器 `mlu-baseline`。版本与
`configs/env/mlu590.lock.yaml` 不完全一致，因此只支持兼容性结论；
锁定环境复验和第二次独立性能复跑仍需执行。
