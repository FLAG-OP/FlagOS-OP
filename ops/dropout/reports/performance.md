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
