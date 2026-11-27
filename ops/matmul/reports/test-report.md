# 测试报告: matmul（2026-11-25/26, 真机 卡 6）

## 范围矩阵

| 层级 | 断言要点 | 结果 |
|---|---|---|
| kernel | 多 shape（含非整倍数尾块）× 三 dtype × 精度/哨兵/性能 | ✅ {'ok': True, 'max_rel_err': 0.007812, 'sentinel': 'deterministic+sensitive', 'latency_ms': 1.0395, 'native_latency_ms':  |
| op | 注册/拦截/counter + 精度不变形 + 反向 + 回退边界 | ✅ {'ok': True, 'registered': 'aten::matmul@AutogradCUDA', 'intercepted_calls': 7, 'inference_mode_note': 'inference_mode 命 |
| framework | mini consumer 基线/注入双跑（恒等/梯度/调用计数） | ✅ {'ok': True, 'consumer': 'mini-attention (QK^T/AV via matmul, bf16)', 'out_diff': 0.0, 'continuation_match': True, 'grad |
| 黄金 | CPU 权威黄金真机校验 | ✅ native: matmul: native @ cuda:6 30 / 30 PASS, worst=7.692e-03 |

## op 层拦截证据

```
{'ok': True, 'registered': 'aten::matmul@AutogradCUDA', 'intercepted_calls': 7, 'inference_mode_note': 'inference_mode 命中 0（known-issues #19.3 边界）'}
```

## 覆盖明细

- shape: 非 BLOCK 整倍数（5000/5120/4097/1023/3072）+ 边界（0 维/单行/空）
- dtype: fp32 / fp16 / bf16
- 特殊值: 负尾部 / 近零 / 非有限（parity 锚定官方, 见 gen_golden 注）/ 特征值
- 哨兵: 同输入确定性 + 输入敏感（#11 静默 no-op / 不写输出类缺陷）
- 2026-11-27 复审回归: PPU 对照 CPU autograd，
  `(2,3,4)@(4,)/(4,)@(2,4,5)/(3,4)@(2,4,5)` 输入/权重梯度最大误差
  `4.7684e-7`。
