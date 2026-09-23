# embedding 测试报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-23 |
| 设备 | p800-kunlunxin，`cuda:1` |
| 环境 | Torch 2.9.0+cu129 / torch_xmlir / Triton 3.0 |
| 结论 | 通过 |

## 范围

| 层级 | 覆盖 | 结果 |
|---|---|---|
| kernel forward | 1D/2D/3D、D8/80/512、empty、int32、padding | ✅ |
| kernel backward | FP32/FP16/BF16 × dense/scale_freq × duplicate/padding | ✅ |
| 黄金 | 117 组 | ✅ |
| A1 | 拦截、direct bitwise、autograd | ✅ |
| 应用层 | `nn.Embedding` + MLP forward/backward/greedy | ✅ |
| 边界 | sparse forward 接受、sparse backward / invalid padding 拒绝 | ✅ |

## 命令

```bash
python3 example.py p800-kunlunxin
python3 script/gen_golden.py
python3 script/check_accuracy.py --impl p800 --device cuda:1
python3 script/check_accuracy.py --impl triton --device cuda:1
python3 script/check_accuracy.py --impl native --device cuda:1
pytest -q tests/unit
```

## 结果摘要

- kernel forward: 20/20，max error 0；
- kernel backward: 6/6，max error 0；
- golden: 117/117，max error 0；
- A1: interception count 4，registered path equals direct path bitwise；
- dense grad error 0；
- scale-grad error 0；
- application logits/gradient diff 0；
- greedy continuation match 1.00。
- sparse=True forward 与 dense forward bitwise equal；
- sparse backward显式拒绝。
