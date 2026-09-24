# 性能分册: empty_strided

## 口径

分配算子，仅测单次分配延迟；短采样 warmup 20 + iters 100 + MLU 同步。
逐次分配触发分配器池增长（known-issues #10），仅可用性采样，不作门禁。

```bash
python3 script/bench_perf.py --device mlu
```

## 结果（MLU590-M9, 8192² fp32）

| 实现 | 延迟 |
|---|---|
| torch 级 | 0.0135ms |
| reference（原生） | 同量级 |
