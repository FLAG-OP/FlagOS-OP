# 测试报告: softmax（2026-11-25/26, 真机 卡 6）

## 范围矩阵

| 层级 | 断言要点 | 结果 |
|---|---|---|
| kernel | 多 shape（含非整倍数尾块）× 三 dtype × 精度/哨兵/性能 | ✅ {'ok': True, 'max_err': 0.0, 'sentinel': 'deterministic+sensitive', 'latency_ms': 61.6605, 'native_latency_ms': 61.6609, |
| op | 注册/拦截/counter + 精度不变形 + 反向 + 回退边界 | ✅ {'ok': True, 'registered': 'aten::softmax@AutogradCUDA', 'intercepted_calls': 12, 'inference_mode_note': 'inference_mode |
| framework | mini consumer 基线/注入双跑（恒等/梯度/调用计数） | ✅ {'ok': True, 'consumer': 'mini-attention softmax(scores) (bf16)', 'out_diff': 0.0, 'continuation_match': True, 'grad_max |
| 黄金 | CPU 权威黄金真机校验 | ✅ native: softmax: native @ cuda:6 24 / 24 PASS, worst=3.576e-07 |

## op 层拦截证据

```
{'ok': True, 'registered': 'aten::softmax@AutogradCUDA', 'intercepted_calls': 12, 'inference_mode_note': 'inference_mode 命中 0（known-issues #19.3 边界）'}
```

## 覆盖明细

- shape: 非 BLOCK 整倍数（5000/5120/4097/1023/3072）+ 边界（0 维/单行/空）
- dtype: fp32 / fp16 / bf16
- 特殊值: 负尾部 / 近零 / 非有限（parity 锚定官方, 见 gen_golden 注）/ 特征值
- 哨兵: 同输入确定性 + 输入敏感（#11 静默 no-op / 不写输出类缺陷）
