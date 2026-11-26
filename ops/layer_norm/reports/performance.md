# 性能分册: layer_norm（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.039, "native_impl_GBps": 1720.48, "native_direct_ms": 0.0388, "native_direct_GBps": 1731.22, "reference_ms": 0.5762, "reference_GBps": 116.46}
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.039, "native_impl_GBps": 1721.73, "native_direct_ms": 0.0388, "native_direct_GBps": 1727.61, "reference_ms": 0.5878, "reference_GBps": 114.17}
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.0757, "native_impl_GBps": 1774.08, "native_direct_ms": 0.0754, "native_direct_GBps": 1780.22, "reference_ms": 0.439, "reference_GBps": 305.75}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
