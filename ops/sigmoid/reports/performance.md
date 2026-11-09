# aten.sigmoid 性能报告

> wt-2026-11-09-sigmoid-p800 · wt <liyanze1560495@hotmail.com>

## 1. 测试口径

| 项 | 值 |
|---|---|
| 设备 | Kunlunxin P800 OAM 96 GiB，物理 XPU 2 |
| 软件 | PyTorch 2.9.0、Triton 3.0.0、FlagGems 5.0.0、XPU-RT 5.0.21 |
| 实现 | 自研 Triton / FlagGems / 原生 ATen |
| 矩阵 | 4 shape × 3 dtype × 3 实现 = 36 个独立子进程 |
| 计时 | 每项先编译，warmup 20、计时 100、前后同步 |

`自研/原生` 小于 1 才表示自研更快。

## 2. 三方延迟

| case | shape | dtype | 自研 ms | FlagGems ms | 原生 ms | 自研/原生 |
|---|---|---|---:|---:|---:|---:|
| tail_hidden | 1×14336 | fp16 | 0.051190 | 0.041866 | 0.019210 | 2.665× |
| tail_hidden | 1×14336 | bf16 | 0.049970 | 0.043415 | 0.011686 | 4.276× |
| tail_hidden | 1×14336 | fp32 | 0.052726 | 0.046549 | 0.019553 | 2.697× |
| small_activation | 64×1024 | fp16 | 0.049493 | 0.040419 | 0.023780 | 2.081× |
| small_activation | 64×1024 | bf16 | 0.058088 | 0.041161 | 0.019554 | 2.971× |
| small_activation | 64×1024 | fp32 | 0.056742 | 0.048821 | 0.020066 | 2.828× |
| transformer | 2×64×4096 | fp16 | 0.050435 | 0.049617 | 0.019240 | 2.621× |
| transformer | 2×64×4096 | bf16 | 0.049383 | 0.041907 | 0.019026 | 2.596× |
| transformer | 2×64×4096 | fp32 | 0.050004 | 0.040520 | 0.019305 | 2.590× |
| throughput | 2048×2048 | fp16 | 0.082352 | 0.082268 | 0.055593 | 1.481× |
| throughput | 2048×2048 | bf16 | 0.077650 | 0.077806 | 0.067039 | 1.158× |
| throughput | 2048×2048 | fp32 | 0.058188 | 0.058258 | 0.048857 | 1.191× |

## 3. Dispatch 开销

| 直调 | A1 注册 | 增量 | 相对增量 |
|---:|---:|---:|---:|
| 47.970 μs | 55.877 μs | 7.908 μs | 16.5% |

## 4. 中央性能入口

`perf_run.py --device p800-liyanz-local --pattern sigmoid` 记录 `ops.sigmoid.triton=0.058 ms`、`577.26 GB/s`。当前不存在历史 baseline，因此 `perf_compare.py` 给出 0 FAIL、0 WARN、1 NEW；NEW 仅表示成功入表。

## 5. 分析

- 本次正式 5.0 结果中，自研 12 个组合均慢于原生，不能声明普遍加速。
- 大张量显著缩小差距，`2048×2048 bf16` 慢 15.8%；小尺寸由约 49～58 μs 自研启动固定成本主导。
- 自研仅在 throughput bf16/fp32 两项略快于 FlagGems 5.0，原生仍是当前最佳基线。
- 后续优化应先测安全的 `numel/dtype` 原生路由，或降低 P800 pointwise launch 成本；不得沿用旧 5.3 环境中曾出现的加速数字。

原始 JSON：服务器 `/data2/liyanz/results/sigmoid-p800-gems5-matrix-final.json`（生成物不入库）。
