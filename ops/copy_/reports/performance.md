# 性能分册: copy_

## 1. 口径

8192×8192 fp32 同形原地拷贝；warmup 20 + iters 100 + `torch.mlu.synchronize`。
用例登记 `common/perf_registry.py`（`ops.copy_`）。

```bash
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.copy_
```

## 2. 结果（MLU590-M9）

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton | 0.242 ms | 2215 GB/s | 1.00x |
| torch 级 | 0.237 ms | 2261 GB/s | 1.02x |
| reference（原生） | **0.237 ms** | 2261 GB/s | 1.02x |

- 内存带宽受限，自研与原生持平（差 ~2%）。
- 2D 转置型（如 dst 连续、src 转置）走 `_copy_2d_swiz` 分块，避免逐元素
  gather（与 contiguous 的优化同源）。

## 3. 基线

入库基线 `ops.copy_.triton` 写入
[perf/baselines/cambricon.json](../../../perf/baselines/cambricon.json)。
