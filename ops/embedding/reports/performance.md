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
| dispatch benchmark | `script/bench_dispatch.py` |

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

### A1 dispatch overhead

`aten::embedding` native、direct backend、A1 注册路径分别在独立子进程测量，
每次调用均读取一个输出元素强制完成。原始数据见
[dispatch_p800-kunlunxin.json](dispatch_p800-kunlunxin.json)。

| mode | median | p20 | p80 |
|---|---:|---:|---:|
| native `aten::embedding` | 0.0489ms | 0.0485ms | 0.0494ms |
| direct backend | 0.0749ms | 0.0743ms | 0.0759ms |
| A1 `F.embedding` | 0.0730ms | 0.0674ms | 0.0898ms |

A1 − native ≈ **0.0241ms**；A1 与 direct backend 基本同档
（本轮 A1 median 略低，差异在同步读回噪声内）。

## 3. 分析

1. 生产路径与 native embedding 在 16k 以上基本持平；
2. 1k 小 shape 有 Python wrapper / 完成守卫开销；
3. Triton gather 正确但慢约 3.5-102x，不作为生产实现；
4. inverse-frequency fallback 填补 XPU native 缺口，而不是更快路径。

### FLAG-OP/gatherFIX 对照

[FLAG-OP/gatherFIX](https://github.com/FLAG-OP/gatherFIX) 修复的是
Ascend `torch.gather` 非连续 index 的正确性。为确认它与 embedding
row lookup 的差异，`script/probe_gatherfix.py` 将其 stride-aware rank-5
算法特化到 P800 `weight[index]`：

| shape | native embedding | ours/index_select | gatherFIX 特化 |
|---|---:|---:|---:|
| 1k×D128 | 0.0386ms | 0.0392ms | 0.4213ms |
| 16k×D128 | 0.0705ms | 0.0738ms | 3.2452ms |
| 131k×D128 | 0.2696ms | 0.3595ms | 22.6192ms |
| 16k×D512 | 0.0636ms | 0.0763ms | 11.3235ms |
| vocab128k 16k×D128 | 0.0525ms | 0.0535ms | 3.3008ms |

精度为 0 error；性能为 native 的 **10.9-177.9x 延迟**。结论：gatherFIX
解决的是 Ascend correctness，不改变当前 P800 production 选择。

复现：

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
python3 script/bench_dispatch.py --device cuda:1 \
  --json-out reports/dispatch_p800-kunlunxin.json
```
