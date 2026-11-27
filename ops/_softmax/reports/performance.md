# 性能分册: _softmax（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.1585, "native_impl_GBps": 1032.82, "native_direct_ms": 4.1586, "native_direct_GBps": 1032.79, "reference_ms": 27.7738, "reference_GBps": 154.64}
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.1666, "native_impl_GBps": 1030.82, "native_direct_ms": 4.1657, "native_direct_GBps": 1031.03, "reference_ms": 28.7247, "reference_GBps": 149.52}
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.5578, "native_impl_GBps": 1884.65, "native_direct_ms": 4.7949, "native_direct_GBps": 1791.48, "reference_ms": 19.8607, "reference_GBps": 432.51}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
