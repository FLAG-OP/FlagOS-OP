# 测试报告: _softmax（2026-11-25/26, 真机 卡 6）

## 范围矩阵

| 层级 | 断言要点 | 结果 |
|---|---|---|
| kernel | 多 shape（含非整倍数尾块）× 三 dtype × 精度/哨兵/性能 | ✅ {'ok': True, 'max_err': 4.768e-07, 'sentinel': 'deterministic+sensitive', 'latency_ms': 61.6563, 'GBps': 557.28} |
| op | 注册/拦截/counter + 精度不变形 + 反向 + 回退边界 | ✅ {'ok': True, 'registered': 'aten::_softmax@AutogradCUDA', 'intercepted_calls': 4, 'public_softmax_hits_primitive_registr |
| framework | mini consumer 基线/注入双跑（恒等/梯度/调用计数） | ✅ {'ok': True, 'consumer': 'mini-attention with aten::_softmax (bf16)', 'out_diff': 0.0, 'continuation_match': True, 'grad |
| 黄金 | CPU 权威黄金真机校验 | ✅ native: _softmax: native @ cuda:6 24 / 24 PASS, worst=6.104e-05 |

## op 层拦截证据

```
{'ok': True, 'registered': 'aten::_softmax@AutogradCUDA', 'intercepted_calls': 4, 'public_softmax_hits_primitive_registry': 1, 'inference_mode_note': 'inference_mode 命中 0'}
```

## 覆盖明细

- shape: 非 BLOCK 整倍数（5000/5120/4097/1023/3072）+ 边界（0 维/单行/空）
- dtype: fp32 / fp16 / bf16
- 特殊值: 负尾部 / 近零 / 非有限（parity 锚定官方, 见 gen_golden 注）/ 特征值
- 哨兵: 同输入确定性 + 输入敏感（#11 静默 no-op / 不写输出类缺陷）
