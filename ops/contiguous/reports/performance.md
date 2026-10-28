# 性能分册: contiguous

## 1. 口径

非连续输入（8192×8192 fp32 转置）→ 连续输出；warmup 20 + iters 100 +
`torch.mlu.synchronize`。用例登记 `common/perf_registry.py`（`ops.contiguous`）。

```bash
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.contiguous
```

## 2. 结果（MLU590-M9）

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton（含 2D swiz） | 0.305 ms | 1762 GB/s | 1.00x |
| torch 级 | 0.254 ms | 2112 GB/s | 1.20x |
| reference（原生） | **0.254 ms** | 2116 GB/s | 1.20x |

## 3. 关键优化

初版 rank-2 strided kernel 逐元素 `i%N`/`i//N` + gather，8192² 转置仅
**21 GB/s（24.7ms）**。端口 `copy_r3` 的 `_contig_2d_swiz` 分块 kernel
（src/dst 最内维都连续 → burst 访存）后 **1762 GB/s（0.305ms）**，
提升 **~80x**，与原生差距从 100x 缩到 1.2x。

## 4. 局限

- rank≥3 非连续仍走逐元素路径，未分块。
- channels_last 等 memory_format 走原生回退（无自研覆盖）。

## 5. 基线

入库基线 `ops.contiguous.triton` 写入
[perf/baselines/cambricon.json](../../../perf/baselines/cambricon.json)。

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；benchmark 使用空闲物理 P800（逻辑 `cuda:1`），run C 建基线、run D 复核，所有用例偏差 ≤5.9%（0.005 ms 级分配/元数据用例绝对差 <2 µs）。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.367 |
| torch | 0.366 |
| triton | 59.244 |

连续性用例 0.367 ms 与原生持平；转置路径 59.24 ms（约 9.1 GB/s），正确性无损，后续可按 XMLIR 重调 2D swizzle。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
