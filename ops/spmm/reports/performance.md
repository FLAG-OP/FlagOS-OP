# 性能分册: spmm（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"M": 8192, "K": 64, "N": 64, "nnz": 5349, "index_add_ms": 0.1195, "index_add_GFLOPS": 5.73, "native_ms": 0.014, "native_GFLOPS": 49.03, "reference_fp64_ms": 0.1261, "reference_fp64_GFLOPS": 5.43}
{"M": 16384, "K": 64, "N": 512, "nnz": 5122, "index_add_ms": 0.1189, "index_add_GFLOPS": 44.13, "native_ms": 0.0538, "native_GFLOPS": 97.43, "reference_fp64_ms": 0.2299, "reference_fp64_GFLOPS": 22.81}
{"M": 4096, "K": 128, "N": 64, "nnz": 10537, "index_add_ms": 0.1178, "index_add_GFLOPS": 11.45, "native_ms": 0.0141, "native_GFLOPS": 95.53, "reference_fp64_ms": 0.1264, "reference_fp64_GFLOPS": 10.67}
```

## 判读

原生 fp32 暖进程可用时更快, 但冷启动失败/fp16 缺口/C++ abort（probes/thead_dense_p0_report.md §三）使其不可作生产路径; index_add 组合为稳定基线。
