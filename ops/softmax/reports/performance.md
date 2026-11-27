# 性能分册: softmax（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.1583, "native_impl_GBps": 1032.87, "native_direct_ms": 4.1583, "native_direct_GBps": 1032.85, "reference_ms": 12.5667, "reference_GBps": 341.77}
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.1656, "native_impl_GBps": 1031.06, "native_direct_ms": 4.1654, "native_direct_GBps": 1031.1, "reference_ms": 13.5116, "reference_GBps": 317.87}
{"case": [8, 32, 2048, 2048], "native_impl_ms": 4.5131, "native_impl_GBps": 1903.35, "native_direct_ms": 4.7894, "native_direct_GBps": 1793.53, "reference_ms": 4.6444, "reference_GBps": 1849.51}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
