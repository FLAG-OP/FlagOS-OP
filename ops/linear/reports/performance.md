# 性能分册: linear（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"M": 4096, "K": 4096, "N": 4096, "native_ms": 1.0308, "native_TFLOPS": 133.33, "torch_ms": 1.0293, "torch_TFLOPS": 133.53, "reference_ms": 4.2389, "reference_TFLOPS": 32.42, "torch_vs_native": 0.999}
{"M": 4096, "K": 4096, "N": 4096, "native_ms": 1.0302, "native_TFLOPS": 133.41, "torch_ms": 1.0272, "torch_TFLOPS": 133.8, "reference_ms": 4.2654, "reference_TFLOPS": 32.22, "torch_vs_native": 0.997}
{"M": 4096, "K": 4096, "N": 4096, "native_ms": 4.0254, "native_TFLOPS": 34.14, "torch_ms": 4.0253, "torch_TFLOPS": 34.14, "reference_ms": 4.0252, "reference_TFLOPS": 34.14, "torch_vs_native": 1.0}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
