# 性能分册: matmul（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"M": 4096, "K": 4096, "N": 4096, "native_impl_ms": 1.0394, "native_impl_TFLOPS": 132.22, "native_direct_ms": 1.0391, "native_direct_TFLOPS": 132.26, "reference_ms": 4.2391, "reference_TFLOPS": 32.42}
{"M": 4096, "K": 4096, "N": 4096, "native_impl_ms": 1.0395, "native_impl_TFLOPS": 132.21, "native_direct_ms": 1.0397, "native_direct_TFLOPS": 132.19, "reference_ms": 4.2657, "reference_TFLOPS": 32.22}
{"M": 4096, "K": 4096, "N": 4096, "native_impl_ms": 4.0248, "native_impl_TFLOPS": 34.15, "native_direct_ms": 4.0248, "native_direct_TFLOPS": 34.15, "reference_ms": 4.0247, "reference_TFLOPS": 34.15}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
