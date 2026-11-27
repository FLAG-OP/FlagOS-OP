# 测试报告: gelu（2026-11-25/26, 真机 卡 6）

## 范围矩阵

| 层级 | 断言要点 | 结果 |
|---|---|---|
| kernel | 多 shape（含非整倍数尾块）× 三 dtype × 精度/哨兵/性能 | ✅ {'ok': True, 'max_err': 0.0009765625, 'sentinel': 'deterministic+sensitive', 'latency_ms': 0.1311, 'native_latency_ms':  |
| op | 注册/拦截/counter + 精度不变形 + 反向 + 回退边界 | ✅ {'ok': True, 'registered': 'aten::gelu@AutogradCUDA', 'intercepted_calls': 9, 'inference_mode_note': 'inference_mode 命中  |
| framework | mini consumer 基线/注入双跑（恒等/梯度/调用计数） | ✅ {'ok': True, 'consumer': 'mini FFN (Linear+GELU+Linear, bf16, 6-step autoreg)', 'logits_diff': 0.0, 'top1_match': True,  |
| 黄金 | CPU 权威黄金真机校验 | ✅ triton: [gelu:triton@cuda:6] 150/150 PASS, worst=9.766e-04；native: [gelu:native@cuda:6] 150/150 PASS, worst=9.766e-04 |

## op 层拦截证据

```
{'ok': True, 'registered': 'aten::gelu@AutogradCUDA', 'intercepted_calls': 9, 'inference_mode_note': 'inference_mode 命中 0（known-issues #19.3 边界）'}
```

## 覆盖明细

- shape: 非 BLOCK 整倍数（5000/5120/4097/1023/3072）+ 边界（0 维/单行/空）
- dtype: fp32 / fp16 / bf16
- 特殊值: 负尾部 / 近零 / 非有限（parity 锚定官方, 见 gen_golden 注）/ 特征值
- 哨兵: 同输入确定性 + 输入敏感（#11 静默 no-op / 不写输出类缺陷）
- 2026-11-27 复审回归: fp64 参考、FlagFix 负尾 4/4 PASS；op 层
  tanh/边界改为按 dtype 容差判卷，不再要求与 fp64 参考逐位一致。
