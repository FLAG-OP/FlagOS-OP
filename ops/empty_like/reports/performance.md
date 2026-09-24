# 性能分册: empty_like

## 口径

分配算子，仅测单次分配延迟；短采样 warmup 20 + iters 100 + MLU 同步。
注意：逐次分配触发分配器池增长（known-issues #10），此处仅可用性采样。

```bash
python3 script/bench_perf.py --device mlu
```

## 结果（MLU590-M9, 8192² fp32）

| 实现 | 延迟 |
|---|---|
| torch 级 | 0.0133 ms |
| reference（原生） | 同量级 |

分配延迟受分配器缓存状态影响大，不作为回归门禁指标。
