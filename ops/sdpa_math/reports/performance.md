# 性能报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> 数据来源: `python3 script/bench_perf.py`（本算子单测）·
> `scripts/perf_run.py --pattern sdpa_math` + `perf_compare.py`（回归门禁）。
> 口径: 短采样 warmup=20 + iters≤100、计时段前后同步、每用例**独立子进程**
> （[#10](../../../docs/known-issues.md)）、读一个输出元素防异步早退。

## 1. 配置

| 项 | 值 |
|---|---|
| shape / dtype | `prefill 1k/2k × D64/D128` · `GQA 1k (H32/8)` · `decode S64` · `tail S100`；NPU=fp16、CPU=fp32 |
| 设备 | Ascend910_9382（`npu:0`）· CPU（128 线程容器） |
| 口径说明 | **同输出契约**才算数：本算子必须物化 `(B,Hq,Sq,Skv)` 概率图，因此基线选同为"返回 `(out, P)`"的原生 **math 后端**；`F.sdpa` 走融合注意力（**不返回 P**）只作参考列 |

## 2. 结果：NPU fp16（`--device npu:0 --register`）

speedup = 原生 math / ours（>1 表示 ours 更快）。数据: [perf_ascend910.json](perf_ascend910.json)。

| shape | 自研 triton | 原生 math | speedup | 自研 A1 包装 | A1 vs 原生 |
|---|---:|---:|---:|---:|---:|
| prefill 1k D64 | 0.528ms | 0.634ms | **1.20x** | 0.618ms | 1.03x |
| prefill 1k D128 | 0.527ms | 0.697ms | **1.32x** | 0.565ms | 1.23x |
| prefill 2k D128 | 1.316ms | 2.692ms | **2.05x** | 1.350ms | 1.99x |
| GQA 1k D128 | 0.804ms | 1.398ms | **1.74x** | 0.836ms | 1.67x |
| decode D128 | 0.287ms | 0.264ms | 0.92x | 0.361ms | 0.73x |
| tail100 D64 | 0.285ms | 0.275ms | 0.97x | 0.362ms | 0.76x |
| **精度代价** | 0（175/175） | 0（131/131） | | 0 | |

参考列——`F.sdpa`（torch_npu 融合注意力，**不返回概率图**，输出契约不同）:
`prefill 1k D128 0.125ms · 2k 0.228ms · GQA 0.154ms · decode 0.054ms`
（[perf_fsdp_context.json](perf_fsdp_context.json)）。它快 4x 左右是因为
不产出 P、且走 FlashAttention 类融合 kernel——**不能**作为本算子的判卷基线，
但说明: 若消费方不需要概率图，应优先用 `F.sdpa`。

## 3. 结果：CPU fp32（`--device cpu`）

数据: [perf_cpu.json](perf_cpu.json)。自研列 = `kernel/torch_level.py`（CPU 交付实现）。

| shape | 自研 torch | 原生 math | speedup | 自研 A1 | A1 vs 原生 |
|---|---:|---:|---:|---:|---:|
| prefill 1k D64 | 149.69ms | 174.29ms | **1.16x** | 122.80ms | 1.42x |
| prefill 1k D128 | 204.06ms | 216.89ms | **1.06x** | 196.17ms | 1.11x |
| prefill 2k D128 | 722.58ms | 1010.66ms | **1.40x** | 726.52ms | 1.39x |
| GQA 1k D128 | 438.80ms | 460.83ms | 1.05x | 380.66ms | 1.21x |
| decode D128 | 1.059ms | 1.316ms | **1.24x** | 1.180ms | 1.12x |
| tail100 D64 | 0.924ms | 1.184ms | **1.28x** | 1.026ms | 1.15x |

## 4. 回归门禁（入库基线）

`python3 scripts/perf_run.py --device ascend910 --pattern sdpa_math --update-baseline`
→ `python3 scripts/perf_compare.py --device ascend910`

| case | 基线 ms | 本次 ms | Δ | 判定 | 附加指标 |
|---|---:|---:|---:|---|---|
| ops.sdpa_math.triton | 0.504 | 0.504 | +0.0% | **OK** | TFLOPS=8.524 |
| ops.sdpa_math.torch | 0.569 | 0.569 | +0.0% | **OK** | TFLOPS=7.549 |
| ops.sdpa_math.reference | 0.892 | 0.892 | +0.0% | **OK** | TFLOPS=4.817 |

**结论: FAIL 0 · WARN 0 · NEW 0**（`examples/` 提供者与本算子无关的
`ops.embedding` 加载失败为既有环境问题，不计入本次门禁）。

> 说明：`ops.sdpa_math` 的 perf 用例已登记进
> [`common/perf_registry.py`](../../../common/perf_registry.py)。基线于
> 2026-09-30 首次入库（算子新增，非环境升级触发）。

## 5. 分析

**为什么大形状快**：两段式 `_probs_kernel` + `_pv_kernel` 对 NPU 的
tile 化更友好，原生 math 后端走 ATen 组合（多轮 `matmul` + `softmax` +
`masked_fill` 中间张量），2k 形状下显存流量差距被放大到 2.05x。

**为什么小形状略慢（0.92-0.97x）**：
1. 本算子必须物化 `(1,16,64,64)` 概率图（额外一次全量写 + 一次全量读），
   `S=64/100` 时计算量小、内存/启动占比高；
2. triton 固定 64×64 tile，`S=64/100` 只有 1-2 个 tile，kernel 利用率低；
3. A1 包装另加约 0.04-0.09ms（`autograd.Function` 的图节点开销），
   对 0.3ms 级调用有感（decode A1 0.361ms vs 原生 0.264ms）。

**优化方向**（未做，非阻塞）：
- 小形状合并 pass1/pass2、减少一次中间张量落地；
- 4k+ 长序列按行块分段 PV，压峰值显存；
- `decode` 场景（`Sq=1`）专用分支，跳过 causal 构造。

**与基线的回归情况**：门禁 OK（Δ=0.0%），无回退。

## 6. 复现

```bash
python3 script/bench_perf.py --device npu:0 --register \
        --json-out reports/perf_ascend910.json                # NPU fp16
python3 script/bench_perf.py --device cpu \
        --json-out reports/perf_cpu.json                      # CPU fp32
python3 ../../scripts/perf_run.py     --device ascend910 --pattern sdpa_math \
        --update-baseline                                    # 采基线（有意动作）
python3 ../../scripts/perf_compare.py --device ascend910      # 门禁（FAIL 0）
```
