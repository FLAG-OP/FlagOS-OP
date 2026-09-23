# embedding 测试报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-23 |
| 设备 | p800-kunlunxin，`cuda:1` |
| 环境 | Torch 2.9.0+cu129 / torch_xmlir / Triton 3.0 |
| 结论 | 通过 |

## 1. 测试范围

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel forward | 1D/2D/3D、D8/80/512、empty、int32、padding | ✅ |
| kernel backward | FP32/FP16/BF16 × dense/scale_freq × duplicate/padding | ✅ |
| 黄金 | 174 组 | ✅ |
| A1 | 拦截、direct bitwise、autograd | ✅ |
| 应用层 | `nn.Embedding` + MLP forward/backward/greedy | ✅ |
| 边界 | sparse forward 接受、sparse backward / invalid padding 拒绝 | ✅ |

## 2. 环境

| 项 | 值 |
|---|---|
| Python | 3.10.18 |
| Torch | 2.9.0+cu129 |
| torch_xmlir | XMLIR--bc1b1dc6f-dev+2026032411 |
| Triton | 3.0.0+03c4c9be |
| 设备 / 卡 | p800-kunlunxin / `cuda:1` |

## 3. 精度结果

复现命令：

```bash
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 script/check_accuracy.py --impl native --device cuda:1
pytest -q tests/unit
```

结果摘要：

- kernel forward: 20/20，max error 0；
- kernel backward: 6/6，max error 0；
- golden: 174/174，max error 0；
- A1: interception count 5，registered path equals direct path bitwise；
- dense grad error 0；
- scale-grad error 0；
- application logits/gradient diff 0；
- greedy continuation match 1.00。
- sparse=True forward 与 dense forward bitwise equal；
- sparse backward显式拒绝。

## 4. 性能结果

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
```

perf gate 结果：`FAIL 0 · WARN 0 · NEW 0`。入库基线中
`ops.embedding.p800.forward` 为 0.051ms，native 对照为 0.052ms。

## 5. 问题与风险

| 风险 | 状态 |
|---|---|
| `scale_grad_by_freq=True` fallback 有额外中间 tensor | 已量化：约 1.96ms vs dense 1.33ms |
| Triton gather 慢 | 仅保留探针，不接生产 |
| sparse backward 未覆盖 | 显式拒绝，符合需求边界 |

## 6. 结论

kernel 层、op 层、应用层、黄金与性能门禁全部通过；建议合入。
