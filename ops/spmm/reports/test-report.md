# 测试报告: spmm（2026-11-25/27, 真机 卡 6）

## 范围矩阵

| 层级 | 断言要点 | 结果 |
|---|---|---|
| kernel | 多 shape（含非整倍数尾块）× 三 dtype × 精度/哨兵/性能 | ✅ {'ok': True, 'max_err': 0.0008873115, 'sentinel': 'deterministic+sensitive', 'nnz': 5330, 'latency_ms': 0.1208, 'GFLOPS' |
| op | 注册/拦截/counter + 精度不变形 + 反向 + 回退边界 | ✅ {'ok': True, 'registered': 'aten::_sparse_mm@AutogradCUDA', 'intercepted_calls': 5, 'dense_mm_untouched': True, 'inferen |
| framework | mini consumer 基线/注入双跑（恒等/梯度/调用计数） | ✅ {'ok': True, 'consumer': '2-layer GCN feature propagation (fp32, 4k nodes)', 'out_diff': 2.2351741790771484e-08, 'propag |
| 黄金 | CPU 权威黄金真机校验 | ✅ native: spmm: native @ cuda:6 24 / 24 PASS, worst=9.533e-04 |

## op 层拦截证据

```
{'ok': True, 'registered': 'aten::_sparse_mm@AutogradCUDA', 'intercepted_calls': 5, 'dense_mm_untouched': True, 'inference_mode_note': 'inference_mode 命中 0（known-issues #19.3 边界）'}
```

## 覆盖明细

- shape: 非 BLOCK 整倍数（5000/5120/4097/1023/3072）+ 边界（0 维/单行/空）
- dtype: fp32 / fp16 / bf16
- 特殊值: 负尾部 / 近零 / 非有限（parity 锚定官方, 见 gen_golden 注）/ 特征值
- 哨兵: 同输入确定性 + 输入敏感（#11 静默 no-op / 不写输出类缺陷）
- 2026-11-27 复审回归: baseline/injected 独立子进程、固定同输入/权重；
  baseline 命中 0，injected 10 次；模型输出 diff `2.2352e-8`，
  2 步传播 diff `5.8208e-10`；`w1/w2` grad diff 分别
  `1.0914e-11` / `2.1828e-11` 且 finite。
- fp16 反向溢出反例（`2*60000 + 2*(-60000)`）升 fp32 归约后
  `d_vals=0`，无 NaN。
