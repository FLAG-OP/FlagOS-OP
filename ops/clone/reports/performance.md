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

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；benchmark 使用空闲物理 P800（逻辑 `cuda:1`），run C 建基线、run D 复核，所有用例偏差 ≤5.9%（0.005 ms 级分配/元数据用例绝对差 <2 µs）。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.245 |
| torch | 0.245 |
| triton | 0.478 |

P800 上自研 Triton 拷贝 0.478 ms / 原生 0.245 ms（约 0.51x，约 1.12 TB/s vs 2.19 TB/s）；功能与黄金无误，性能瓶颈在 XMLIR 下 Triton 1D copy 的向量化/启动开销，后续可调 block/向量宽度。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
