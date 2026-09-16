# SDPA Triton vs 原生 CANN 性能差距根因分析

日期: 2026-09-16 · 环境: Ascend 910 ×8 (CANN 9.0.0) / triton 3.5.1 (BiShengIR) / torch_npu 2.10.0
实验脚本: `script/perf_explore.py` `script/perf_explore2.py`（日志同名 .log）

## 一句话结论

差距的主因不在我们的 kernel 结构，而在 **Triton→BiShengIR 栈的 GEMM 效率天花板**：
同形状纯 matmul（无 softmax、BM=BN=128）Triton 就比 CANN 优化 GEMM 慢 **5.2x**
（0.057 vs 0.011 ms）。SDPA 的 12x 差距 = GEMM 栈差距（~5x）× SDPA 结构放大
（online softmax 向量段 + 64×64 tile 上限，~2.4x）。

## 实验证据链

### A. tile/warps/stages 扫描（S=2048 D=128 H=16 causal fp16）

| 配置 | 结果 |
|---|---|
| BM/BN 64×64（现配置） | 2.767 ms，6.2 TFLOPS，12.2x native |
| warps 4↔8，stages None/2/3 | **完全无影响**（2.766-2.768 ms） |
| BM 或 BN > 64 全部组合（含 128×32/256×32） | **MLIR 编译失败**（UB 溢出） |
| 64×32 | 5.888 ms（更差——迭代次数×2，向量段开销×2） |
| LOAD_KT（消 tl.trans） | 编译失败未能对比（与 128 tile 同因） |

warps/stages 零敏感 + 64×32 慢一倍 ⇒ 时间不在 cube 计算的调度上，
而在**每次 K 迭代的固定开销**（softmax 向量段 + 小 dot 的低 cube 效率）。

### B. 纯 matmul 天花板（同形状 2048×128×2048 fp16，BM=BN=128）

| 实现 | 耗时 | 效率 |
|---|---|---|
| Triton `tl.dot` matmul | 0.057 ms | 9.5 GFLOPS/call |
| torch.matmul（CANN GEMM） | 0.011 ms | **5.2x** 快 |

16 heads × 2 GEMM 按 Triton 效率推算 = 1.8 ms ≈ 我们 SDPA 的 65%
——我们 kernel 的 softmax/online 开销约 35%，结构已接近该栈 GEMM 效率下的合理水位。

### C. 带宽与利用率核算

- 本机实测（npu-smi -t common）: Ascend910_9382，**24 AI cores/卡** @1800MHz
  （非经典 910 的 32 核；按 4096 MAC/cycle/core 折算 fp16 峰值 ~350 TFLOPS 量级，
  精确分母以官方规格为准——本节利用率数字按此口径理解）
- ours 6.2 TFLOPS ≈ 峰值 2-3%；native 75.5 TFLOPS ≈ 峰值 21-35%
- K/V 名义流量被 grid_m=32 放大到 537 MB/次 → 需 194 GB/s HBM（峰值 ~1200）——
  **带宽不是瓶颈**，cube 利用率才是
- native 只要 L2 命中 K/V（17 MB 全量，L2 可容纳）仅 74 GB/s HBM

### D. dtype × 差距（S=2048 D=128 H=16 causal）

| dtype | native | ours | 差距 |
|---|---|---|---|
| fp16 | 0.228 ms | 2.760 ms | 12.1x |
| bf16 | 0.233 ms | 2.770 ms | 11.9x |
| fp32 | 0.345 ms | 2.968 ms | 8.6x（ieee 点积，无 tf32 捷径，native 差距反小）|

## 架构层解释（对照公开资料）

1. **解耦引擎模型**: 910 的 cube(AIC)/vector(AIV)/MTE 之间无寄存器直通，
   online softmax 的 exp/max/sum 向量段天然把两次 cube 调用隔断。
   CANN 原生 FA（AscendC 手写）用 L0A/L0B multi-buffer + MTE 流水把
   数据搬运与计算重叠；Triton 栈对此的控制粒度有限（num_stages 实测无效）。
2. **UB 硬约束**: 192KB/核。D=128 时 128-M tile 的 Q/acc( fp32 ) + K/V 块 +
   softmax 状态超限，BiShengIR 直接拒绝编译（mlir ub overflow 报错），
   只能跑 64×64——而 CANN FA 用分阶段驻留（Q 常驻 + K/V 分块流式）绕开。
3. **fractal 布局**: CANN GEMM 输入走 16×16 fractal 摆放直达 cube；
   Triton 的 ND 布局需编译器插入重排，小 dot 下重排开销占比高。

## 我们 kernel 可改进但受栈约束的部分（诚实清单）

- [x] 已做: 固定 tile 免 autotune、尾块安全、fp32 ieee、P-cast 同 dtype dot
- [ ] LOAD_KT 变体（消 tl.trans）: 64×64 下当前编译失败，待 BiShengIR 支持
      或改为 host 侧预转置 K（多一次物化，需 A/B 验证净收益）
- [ ] GQA 场景 K/V 复用: 同 h_kv 组的 4 个 M 块重复读 K/V，可按 KV 块外层
      循环重排（FA2 风格）——但 910 上外层 KV 循环会加剧 UB 压力
- [x] 结论: Triton 级在该栈上的合理目标 ≈ 1.8 ms（纯 GEMM 推算下限），
      再往下需要 AscendC 硬件级实现（超出本任务"第二层"范围，符合预期）
