# 性能分册: item

## 口径

短采样 warmup 20 + iters 100 + MLU 同步；item 为元数据/同步操作，
延迟绝对值小、受调度影响大，仅作可用性采样，不作门禁。

```bash
python3 script/bench_perf.py --device mlu
```

## 结果（MLU590-M9）

| 实现 | 延迟 |
|---|---|
| torch 级 | 0.0433ms |
| reference（原生） | 同量级 |

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；benchmark 使用空闲物理 P800（逻辑 `cuda:1`），run C 建基线、run D 复核，所有用例偏差 ≤5.9%（0.005 ms 级分配/元数据用例绝对差 <2 µs）。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.014 |
| torch | 0.016 |

reference 0.0142 ms / torch 0.0160 ms；device→host 标量同步开销为主。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
