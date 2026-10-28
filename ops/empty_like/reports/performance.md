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

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；benchmark 使用空闲物理 P800（逻辑 `cuda:1`），run C 建基线、run D 复核，所有用例偏差 ≤5.9%（0.005 ms 级分配/元数据用例绝对差 <2 µs）。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.0046 |
| torch | 0.0068 |

reference 0.0046 ms / torch 0.0068 ms；stride 由 meta 推导，设备无关。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
