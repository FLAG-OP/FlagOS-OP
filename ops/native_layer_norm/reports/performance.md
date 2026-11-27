# 性能分册: native_layer_norm（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.039, "native_impl_GBps": 1721.28, "native_direct_ms": 0.0388, "native_direct_GBps": 1728.83, "reference_ms": 0.5713, "reference_GBps": 117.47}
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.0392, "native_impl_GBps": 1713.4, "native_direct_ms": 0.0388, "native_direct_GBps": 1730.12, "reference_ms": 0.5844, "reference_GBps": 114.83}
{"case": "{'tokens': 4096, 'hidden': 4096}", "native_impl_ms": 0.0756, "native_impl_GBps": 1775.51, "native_direct_ms": 0.0754, "native_direct_GBps": 1780.06, "reference_ms": 0.4388, "reference_GBps": 305.89}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
