# 性能报告: embedding

## 1. 配置

| 项 | 值 |
|---|---|
| 日期 | 2026-09-24 |
| 设备 | p800-kunlunxin / `cuda:1`（§2）；cambricon MLU590 / `mlu:0`（§3） |
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

### A1 dispatch overhead（P800）

`aten::embedding` native、direct backend、A1 注册路径分别在独立子进程测量，
每次调用均读取一个输出元素强制完成。原始数据见
[dispatch_p800-kunlunxin.json](dispatch_p800-kunlunxin.json)。

| mode | median | p20 | p80 |
|---|---:|---:|---:|
| native `aten::embedding` | 0.0489ms | 0.0485ms | 0.0494ms |
| direct backend | 0.0749ms | 0.0743ms | 0.0759ms |
| A1 `F.embedding` | 0.0730ms | 0.0674ms | 0.0898ms |

> ⚠️ **A1 行作废待复测**（2026-09-24 发现）：当时 `bench_dispatch.py` 裸调
> `register_a1(...)`，返回的 `torch.library.Library` 无引用即被 GC 并
> **反注册**，该子进程实测的是未注册状态。因此原「A1 − native ≈ 0.0241ms、
> A1 与 direct 同档」的结论不可用；native / direct 两行不受影响，仍然有效。
> 本机无 `torch_xmlir`，P800 无法重测，需在 P800 环境复跑
> `script/bench_dispatch.py`。修复与 MLU 实测见
> [cambricon.md](cambricon.md) §7。

## 3. cambricon（MLU590）

数据：[perf_fp16_cambricon.json](perf_fp16_cambricon.json)、
[dispatch_cambricon.json](dispatch_cambricon.json)（warmup=20，iters=100，
每次读回一个输出元素强制完成；speedup = native/ours，**>1 表示 ours 更快**）。

### Forward（fp16）

| shape | ours | native | Triton gather | ours/native |
|---|---:|---:|---:|---:|
| 1k×D128 | 0.0681ms | 0.0445ms | 0.2995ms | 0.654x |
| 16k×D128 | 0.0612ms | 0.0451ms | 2.3029ms | 0.736x |
| 131k×D128 | 0.0819ms | 0.0662ms | 失败（grid 65536>65535） | 0.809x |
| 16k×D512 | 0.0550ms | 0.0617ms | 10.5459ms | **1.123x** |
| vocab128k 16k×D128 | 0.0525ms | 0.0404ms | 2.3306ms | 0.769x |

### Backward（16k×D128，fp16）

| mode | ours | native | ours/native |
|---|---:|---:|---:|
| dense | 0.2651ms | 0.2497ms | 0.942x |
| scale_grad_by_freq | 0.2944ms | 0.2796ms | 0.950x |

### 仓库 perf gate

```text
ops.embedding.cambricon.forward   0.060ms · 142.8 GB/s
ops.embedding.native.forward      0.045ms · 187.9 GB/s
FAIL 0 · WARN 0 · NEW 0
```

`perf/baselines/cambricon.json` 共 33 条 case（新增本算子 2 条）。

### A1 dispatch overhead（MLU，修复后）

| mode | median | p20 | p80 |
|---|---:|---:|---:|
| native `aten::embedding` | 0.0450ms | 0.0448ms | 0.0454ms |
| direct backend | 0.0600ms | 0.0596ms | 0.0608ms |
| A1 `F.embedding` | 0.1041ms | 0.1035ms | 0.1054ms |

A1 − native ≈ **+0.059ms**、A1 − direct ≈ **+0.044ms**；2026-09-25 复测 3 轮
A1 − native = 0.059 / 0.060 / 0.061ms，结论稳定。MLU 上 A1 比 native 慢
约 1.3 倍（0.045→0.104ms），主要来自 `F.embedding` 的 Python 归一化 +
`autograd.Function` 往返；P800 侧同口径数字待复测。

## 4. 分析

P800：

1. 生产路径与 native embedding 在 16k 以上基本持平；
2. 1k 小 shape 有 Python wrapper / 完成守卫开销；
3. Triton gather 正确但慢约 3.5-102x，不作为生产实现；
4. inverse-frequency fallback 填补 XPU native 缺口，而不是更快路径。

cambricon：

1. 两平台同为「原生 gather 委托」，ours/native 都落在 0.65-1.12x；
   差距主要是 ours 侧的 Python wrapper + 完成守卫，不是 kernel 慢；
2. 唯一稳定超过 native 的是 16k×D512（1.123x）——D 越大，
   `index_select` 的启动开销占比越小；
3. MLU 的 Triton gather 慢 6.7-171x，且 131k indices 直接撞
   `grid=65535` 上限失败，只能当精度对照；
4. `scale_grad_by_freq` 用原生分支代价 +11%（P800 手写补偿 +47%）；
5. A1 路径在 MLU 上比 native 慢 +0.059ms，是本算子在该栈上可量化的
   分发成本；P800 同口径待复测（dispatch 脚本 Library GC 缺陷）。

复现：

```bash
python3 script/bench_perf.py --device cuda:1 --dtype float16 \
  --json-out reports/perf_fp16_p800-kunlunxin.json
python3 scripts/perf_run.py --device p800-kunlunxin --pattern ops.embedding
python3 scripts/perf_compare.py --device p800-kunlunxin
python3 script/bench_dispatch.py --device cuda:1 \
  --json-out reports/dispatch_p800-kunlunxin.json

MLU_VISIBLE_DEVICES=7 python3 script/bench_perf.py --device mlu:0 --dtype float16 \
  --json-out reports/perf_fp16_cambricon.json
MLU_VISIBLE_DEVICES=7 python3 scripts/perf_run.py --device cambricon --pattern ops.embedding
MLU_VISIBLE_DEVICES=7 python3 scripts/perf_compare.py --device cambricon
MLU_VISIBLE_DEVICES=7 python3 script/bench_dispatch.py --device mlu:0 \
  --json-out reports/dispatch_cambricon.json
```
