# 性能分册: dropout

## 1. 口径

8192×8192 fp16，p=0.5，train=True；warmup 20 + iters 100 +
`torch.mlu.synchronize`。用例登记 `common/perf_registry.py`（`ops.dropout`）。

```bash
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.dropout
```

## 2. 结果（MLU590-M9）

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton（Philox） | 2.785 ms | 96 GB/s | 1.00x |
| FlagGems `ops.dropout` | 2.960 ms | 90 GB/s | 0.94x |
| torch 级 | 0.563 ms | 476 GB/s | 4.95x |
| reference（原生） | **0.563 ms** | 476 GB/s | 4.95x |

## 3. 分析

- 该 shape 下瓶颈是 **RNG 生成**而非访存：原生使用硬件优化 RNG，
  Triton 用 `tl.philox` 计数器 RNG，逐元素生成 32-bit 随机数。
- 单独计时显示 kernel 本体 2.785ms，`_philox_seed_offset` 仅 ~0.04µs/次，
  可忽略——瓶颈在 kernel 内。
- FlagGems `dropout` 采用同机制，实测 2.96ms，自研略快 ~6%。
- 优化方向: 提高 UNROLL、一次 philox 覆盖更多元素、减少 mask 分支。

## 4. 基线

入库基线 `ops.dropout.triton` 写入
[perf/baselines/cambricon.json](../../../perf/baselines/cambricon.json)。

## P800（p800-kunlunxin）复验（2026-10-26）

环境：torch 2.9.0+cu129 / torch_xmlir 1.0.0.1 / Triton 3.6.0 / FlagGems 4.2.1rc0；benchmark 使用空闲物理 P800（逻辑 `cuda:1`），run C 建基线、run D 复核，所有用例偏差 ≤5.9%（0.005 ms 级分配/元数据用例绝对差 <2 µs）。口径与 MLU 相同（warmup 20 + iters 100 + 设备同步）。

| 实现 | 延迟 (ms) |
|---|---:|
| reference | 0.385 |
| torch | 0.385 |
| triton | 0.385 |

自研 Triton 0.385 ms = 原生 0.385 ms（1.00x）；P800 随机分支委托原生，channels_last 已还原。

基线已写入 `perf/baselines/p800-kunlunxin.json`；`python3 scripts/perf_compare.py --device p800-kunlunxin` → **OK**（FAIL 0 / WARN 0 / NEW 0）。
