# 性能分册: gelu（2026-11-25/26, 真机 卡 6）

## 测量口径

warmup 20 + iters ≤100（known-issues #10 分配器陷阱: 不用长循环）·
计时区段前后 synchronize · 多用例独立口径 · 数字可由
`script/bench_perf.py` 复现（JSON: reports/perf_*.json）。

## 结果（原始 JSON 为准）

```
{"triton_ms": 0.0749, "triton_GBps": 895.4, "native_ms": 0.0343, "native_GBps": 1955.25, "torch_ms": 0.0343, "torch_GBps": 1959.02, "reference_ms": 0.8809, "reference_GBps": 76.19, "triton_vs_native": 0.458}
{"triton_ms": 0.0764, "triton_GBps": 878.75, "native_ms": 0.035, "native_GBps": 1919.2, "torch_ms": 0.0349, "torch_GBps": 1923.57, "reference_ms": 0.8963, "reference_GBps": 74.87, "triton_vs_native": 0.458}
{"triton_ms": 0.0764, "triton_GBps": 1756.13, "native_ms": 0.0692, "native_GBps": 1940.82, "torch_ms": 0.0692, "torch_GBps": 1940.03, "reference_ms": 0.7372, "reference_GBps": 182.07, "triton_vs_native": 0.906}
```

## 判读

对照口径详见 reports/perf_*.json; 复现命令见 README。
`1.088×` 仅适用 **8192×8192 半精度**：fp16=1.102×，bf16=1.098×；
fp32 同形状为 0.993×。1×5000、128×5120 等小形状受 launch 开销主导
（约 0.147–0.149×），4096×4096 全 dtype 慢于 native（0.458–0.906×）；
这些目标形状应走 native。
