# 性能分册: type_as

## 1. 口径

- 短采样: warmup 20 + iters 100 + `torch.mlu.synchronize`（防 #10 分配器池增长）
- 形状: 8192×8192，输入 fp16 → 输出 fp32
- 用例已登记 `common/perf_registry.py`（provider `ops.type_as`），
  每用例独立子进程（`scripts/perf_run.py`）

```bash
python3 script/bench_perf.py --device mlu --impl triton
python3 scripts/perf_run.py --device cambricon --pattern ops.type_as --update-baseline
python3 scripts/perf_compare.py --device cambricon
```

## 2. 结果（MLU590-M9）

| 实现 | 延迟 | 有效带宽 | 相对自研 |
|---|---|---|---|
| **自研 Triton** | 0.190 ms | 2122 GB/s | 1.00x |
| PyTorch 原生 `.to` | **0.175 ms** | 2300 GB/s | 1.09x 更快 |
| FlagGems `ops.to_copy` | 0.317 ms | 1270 GB/s | 0.60x |

有效带宽 = (读 self 2B + 写 out 4B) × 67.1M / t。cast 为纯带宽受限算子。

## 3. 基线

- 入库基线: [perf/baselines/cambricon.json](../../../perf/baselines/cambricon.json)
  （case `ops.type_as.triton`）
- 门禁: 20% WARN / 30% FAIL；基线绑定 git commit + 环境指纹，
  日常改动不允许更新基线。

## 4. 分析

- 自研 kernel 慢于原生约 9%：原生 `.to` 使用 runtime 的向量化设备码，
  自研为固定 BLOCK（连续侧 65536）+ masked load/store。
- 明显快于 FlagGems `to_copy`（0.317ms）：后者走 `pointwise_dynamic`
  生成的通用索引，额外开销更大。
- 优化方向：连续侧去 mask（pad 到 BLOCK 整倍数）、提高向量化程度；
  非 dense 窄→宽改两阶段 kernel 去掉 `out.copy_` 兜底。

## 5. 复现的一处修复

`common/perf.py` 的 `_sync` 原本只 `torch.cuda.synchronize()`；在 MLU
上不同步，实测出现 25 TB/s 的假数据。已补 `torch.mlu.synchronize()`，
上表为修复后数据。
