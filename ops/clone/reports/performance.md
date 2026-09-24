# 性能分册: clone

## 1. 口径

短采样 warmup 20 + iters 100 + `torch.mlu.synchronize`；8192×8192 fp32。
用例登记 `common/perf_registry.py`（provider `ops.clone`）。

```bash
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.clone
python3 scripts/perf_compare.py --device cambricon
```

## 2. 结果（MLU590-M9）

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton | 0.242 ms | 2214 GB/s | 1.00x |
| torch 级 | 0.238 ms | 2256 GB/s | 1.02x |
| reference（原生 clone） | **0.238 ms** | 2257 GB/s | 1.02x |

- FlagGems 无 `clone` 算子，无第三方数据。
- clone 为纯带宽受限，自研与原生持平（差 ~2%）。
- 优化空间有限；若追求极限可尝试更大 BLOCK/向量化。

## 3. 基线

入库基线 `ops.clone.triton` 写入
[perf/baselines/cambricon.json](../../../perf/baselines/cambricon.json)，
门禁 20% WARN / 30% FAIL。
