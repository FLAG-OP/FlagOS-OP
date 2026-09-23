# 性能报告: embedding

## 1. 配置

| 项 | 值 |
|---|---|
| 日期 | 2026-09-24 |
| 设备 | p800-kunlunxin / `cuda:1` |
| dtype | FP16 |
| 计时口径 | 每次读取一个输出元素，强制 XMLIR kernel 完成 |
| operator benchmark | `script/bench_perf.py` |
| perf gate | `scripts/perf_run.py` / `scripts/perf_compare.py` |

## 2. 结果

### Forward

| shape | ours | native embedding | Triton gather | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0680ms | 0.0358ms | 0.2403ms | 0.526x |
| 16k×D128 | 0.0514ms | 0.0494ms | 2.2206ms | 0.963x |
| 131k×D128 | 0.1665ms | 0.1530ms | 16.9373ms | 0.919x |
| 16k×D512 | 0.0616ms | 0.0548ms | 8.7184ms | 0.889x |
| vocab128k 16k×D128 | 0.0514ms | 0.0515ms | 2.2458ms | 1.001x |

### Backward

16k indices × D128：

| mode | ours | native |
|---|---:|---:|
| dense | 1.3306ms | 1.2993ms |
| inverse-frequency | 1.9562ms | XPU native 不支持 |

### 仓库 perf gate

注册用例：

```text
ops.embedding.p800.forward
ops.embedding.native.forward
```

入库基线：

| case | latency | 带宽估算 |
|---|---:|---:|
| ops.embedding.p800.forward | 0.051ms | 167.9 GB/s |
| ops.embedding.native.forward | 0.052ms | 165.3 GB/s |

门禁结论：

```text
FAIL 0 · WARN 0 · NEW 0
```

## 3. 分析

1. 生产路径与 native embedding 在 16k 以上基本持平；
2. 1k 小 shape 有 Python wrapper / 完成守卫开销；
3. Triton gather 正确但慢约 3.5-102x，不作为生产实现；
4. inverse-frequency fallback 填补 XPU native 缺口，而不是更快路径。

复现：

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
```
