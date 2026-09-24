# 性能分册: detach

## 口径

短采样 warmup 20 + iters 100 + MLU 同步；detach 为元数据/同步操作，
延迟绝对值小、受调度影响大，仅作可用性采样，不作门禁。

```bash
python3 script/bench_perf.py --device mlu
```

## 结果（MLU590-M9）

| 实现 | 延迟 |
|---|---|
| torch 级 | 0.0066ms |
| reference（原生） | 同量级 |
