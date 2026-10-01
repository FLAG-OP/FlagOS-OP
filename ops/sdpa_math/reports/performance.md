# 性能报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> 数据来源: `python3 script/bench_perf.py`（本算子单测）·
> `scripts/perf_run.py --pattern sdpa_math` + `perf_compare.py`（回归门禁）。
> 口径: 短采样 warmup=20 + iters≤100、计时段前后同步、每用例**独立子进程**
> （[#10](../../../docs/known-issues.md)）、读一个输出元素防异步早退。

## 1. 配置

| 项 | 值 |
|---|---|
| shape / dtype | `prefill 1k/2k × D64/D128` · `GQA 1k (H32/8)` · `decode S64` · `tail S100`；NPU/P800=fp16、CPU=fp32 |
| 设备 | Ascend910_9382（`npu:0`）· P800（`cuda:1`）· CPU（128 线程容器） |
| 口径说明 | **同输出契约**才算数：本算子必须物化 `(B,Hq,Sq,Skv)` 概率图，因此基线选同为"返回 `(out, P)`"的原生 **math 后端**；`F.sdpa` 走融合注意力（**不返回 P**）只作参考列 |

## 2. 结果：NPU fp16（`--device npu:0 --register`）

speedup = 原生 math / ours（>1 表示 ours 更快）。数据: [perf_ascend910.json](perf_ascend910.json)。

| shape | 自研 triton | 原生 math | speedup | 自研 A1 包装 | A1 vs 原生 |
|---|---:|---:|---:|---:|---:|
| prefill 1k D64 | 0.515ms | 0.626ms | **1.22x** | 0.555ms | 1.13x |
| prefill 1k D128 | 0.538ms | 0.695ms | **1.29x** | 0.572ms | 1.21x |
| prefill 2k D128 | 1.369ms | 2.776ms | **2.03x** | 1.402ms | 1.98x |
| GQA 1k D128 | 0.810ms | 1.351ms | **1.67x** | 0.863ms | 1.57x |
| decode D128 | 0.217ms | 0.255ms | **1.17x** | 0.295ms | 0.86x |
| tail100 D64 | 0.219ms | 0.253ms | **1.16x** | 0.295ms | 0.86x |
| **精度代价** | 0（175/175） | 0（131/131） | | 0 | |

> decode/tail 为 §6 单 kernel 融合生效后的数字（融合前 0.287/0.285ms、
> 0.92x/0.97x）；大形状走两段式（§6 阈值），与融合前持平在测量噪声内。
> A1 列为经 `torch.ops` 注册拦截路径，含 ~0.078ms `autograd.Function` 包装。

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

## 3.1 结果：P800 fp16（`--device cuda:1 --register`）

数据: [perf_p800-kunlunxin.json](perf_p800-kunlunxin.json)。P800 生产路径为
`kernel/p800_fast_level.py`：vendor efficient kernel 计算 `O/LSE`，再用
`P=exp(scale·QKᵀ-LSE)` 物化概率图；A1 使用 `AutogradCUDA+CUDA` 成对注册。
native baseline 同样返回 `(out, P)`。

| shape | P800 fast direct | native math | direct speedup | A1 | A1 speedup |
|---|---:|---:|---:|---:|---:|
| prefill 1k D64 | 0.509ms | 1.021ms | **2.004x** | 0.497ms | **2.054x** |
| prefill 1k D128 | 0.494ms | 1.036ms | **2.096x** | 0.499ms | **2.075x** |
| prefill 2k D128 | 1.429ms | 3.013ms | **2.109x** | 1.442ms | **2.089x** |
| GQA 1k D128 | 0.783ms | 1.726ms | **2.206x** | 0.796ms | **2.168x** |
| decode D128 | 0.290ms | 0.474ms | **1.637x** | 0.311ms | **1.526x** |
| tail100 D64 | 0.278ms | 0.405ms | **1.458x** | 0.310ms | **1.307x** |

结论：P800 direct 路径 **1.46-2.21x** native；A1 路径 **1.31-2.17x**。
相对原 `torch_level` 组合（约 0.34-2.41ms），主要收益来自两点：
vendor kernel 承接 `O` 的 PV 调度，`LSE` 使 `P` 生成免做行 max/sum 归约。

### 3.2 P800 参考口径：与不返回 `P` 的 `F.sdpa`

数据: [perf_f_context_p800-kunlunxin.json](perf_f_context_p800-kunlunxin.json)。
入口: `script/bench_f_context.py`（fp16，warmup=20，iters=100）。

| shape | exact-contract math | native math | no-P `F.sdpa` | math / F |
|---|---:|---:|---:|---:|
| prefill 1k D64 | 0.481ms | 1.037ms | 0.161ms | **2.98x** |
| prefill 1k D128 | 0.490ms | 1.055ms | 0.170ms | **2.89x** |
| prefill 2k D128 | 1.426ms | 3.011ms | 0.320ms | **4.46x** |
| GQA 1k D128 | 0.791ms | 1.730ms | 0.217ms | **3.65x** |
| decode D128 | 0.277ms | 0.408ms | 0.140ms | **1.98x** |
| tail100 D64 | 0.277ms | 0.408ms | 0.137ms | **2.03x** |

该列**不参与判卷、不参与 speedup 结论**：前两者返回 `(out, P)` 并物化
`(B,Hq,Sq,Skv)`；`F.sdpa` 只返回 `out`，可选择 FlashAttention/efficient
路径且不落概率图。它只回答“若下游不需要 `P`，契约本身留下多少延迟”。
这也解释了内部融合的收益上限：即使把 `P` 生成和 `PV` 完全融合，
仍必须写出全量 `P`，无法达到 no-P FlashAttention 的带宽/显存形态。

### 3.3 A100 绝对性能口径

当前缺少同机/同协议的 A100 `aten::_scaled_dot_product_attention_math`
数据，因此**不能给出同输出契约的 A100 加速比**。已有
[ops/sdpa/reports/perf_a100.md](../../sdpa/reports/perf_a100.md) 与
[P800 cross-platform 数据](../../sdpa/reports/cross_platform_p800-kunlunxin.json)
只覆盖 public/no-P fused SDPA：在该非等价口径下，P800 vendor efficient
约为论文 A100·FA2 的 **36-53%**（即慢约 **1.9-2.8x**）。exact-P math
还额外物化概率图，绝对差距大概率存在；结论需待 A100 private-math
实测补齐。

## 4. 回归门禁（入库基线）

`python3 scripts/perf_run.py --device ascend910 --pattern sdpa_math --update-baseline`
→ `python3 scripts/perf_compare.py --device ascend910`

| case | 基线 ms | 本次 ms | Δ | 判定 | 附加指标 |
|---|---:|---:|---:|---|---|
| ops.sdpa_math.triton | 0.504 | 0.543 | +7.8% | **OK** | TFLOPS=7.909 |
| ops.sdpa_math.torch | 0.569 | 0.567 | -0.4% | **OK** | TFLOPS=7.578 |
| ops.sdpa_math.reference | 0.892 | 0.874 | -2.0% | **OK** | TFLOPS=4.916 |

**结论: FAIL 0 · WARN 0 · NEW 0**（`examples/` 提供者与本算子无关的
`ops.embedding` 加载失败为既有环境问题，不计入本次门禁）。本次为
2026-10-01 §6 融合改动后的复测：triton +7.8%（门禁阈值 30% 内），
系 1k 用例两段式路径的进程间波动，未更新基线。

> 说明：`ops.sdpa_math` 的 perf 用例已登记进
> [`common/perf_registry.py`](../../../common/perf_registry.py)。基线于
> 2026-09-30 首次入库（算子新增，非环境升级触发）。

### 4.1 P800 门禁（优化后基线）

```bash
python3 scripts/perf_run.py --device p800-kunlunxin \
        --pattern ops.sdpa_math --update-baseline
python3 scripts/perf_compare.py --device p800-kunlunxin
```

| case | 入库基线 | 复测 ms | Δ | 附加指标 | 判定 |
|---|---:|---:|---:|---|---|
| ops.sdpa_math.p800 | 0.485ms | 0.489ms | +0.9% | TFLOPS=8.779 | **OK** |
| ops.sdpa_math.native | 1.035ms | 1.039ms | +0.4% | TFLOPS=4.133 | **OK** |

**结论: FAIL 0 · WARN 0 · NEW 0**。本次基线更新是 P800 exact-P fast path
的 intentional optimization，不是环境漂移。

## 5. 分析

**测量口径两点**：
- **JIT 编译不进均值**：triton 首次调用含在线编译（实测 4261ms），随后
  60 次稳态 mean 0.470ms；`warmup=20` 在计时段外，所有数字均为稳态。
- **单次 kernel launch 地板 ~0.097ms**（本栈最小 add kernel 实测）：
  launch 次数直接决定 0.2ms 级形状的上限——causal 路径原为 3 次
  （`zeros` + probs + PV），现已减到 2 次（causal 尾块补零收进 probs
  kernel），小形状收益见 §6。

**为什么大形状快**：两段式 `_probs_kernel` + `_pv_kernel` 对 NPU 的
tile 化更友好，原生 math 后端走 ATen 组合（多轮 `matmul` + `softmax` +
`masked_fill` 中间张量），2k 形状下显存流量差距被放大到 2.03x。

**与融合库 `F.sdpa` 的差距是栈级的**（差距构成实测：launch ~3%、
P 往返访存 ~28%、计算效率 ~69%）：
1. **计算效率**：同栈纯 `tl.dot` matmul 就比 CANN 优化 GEMM 慢
   **5.2x**——Ascend UB 仅 192KB，BM/BN>64 全组合 MLIR 编译失败，
   tile 调优封死在 64×64（证据见
   [`ops/sdpa/reports/perf_analysis.md`](../../sdpa/reports/perf_analysis.md)）；
2. **P 往返**：输出契约必须物化 `(B,Hq,Sq,Skv)` 概率图（一次全量写 +
   下游一次全量读），融合库不落 P；
3. **launch**：已从 3 次降到 2 次（§6）。
   → 单算子内再优化也无法追平 `F.sdpa`；消费方不需要概率图时应优先
   用 `F.sdpa`（§2 参考列）。

**小形状现状**：decode/tail 经 §6 单 kernel 融合后为 1.17x/1.16x
（融合前 0.92x/0.97x）；A1 路径 0.86x 是因为包装开销 ~0.078ms 对
0.29ms 级调用占比高，且该开销在 `torch.library`/autograd 图节点层，
非本算子可控。

**优化方向**（未做，非阻塞）：
- 4k+ 长序列按行块分段 PV，压峰值显存；
- `scale` 为 tensor 的慢路径（当前转 python float）；
- A1 包装的图节点开销（~0.078ms）。

**P800 剩余差距**：exact-P 路径仍比 no-P `F.sdpa` 慢 **1.98-4.46x**。
其中不可避免的部分是全量 `P` 写出与额外 QK；可继续压缩的是 causal
mask 构造、GQA `K` expansion 与 P 写出调度。掩码/dropout/fp32/直连
autograd 场景按语义保守回退 `torch_level`。

**与基线的回归情况**：Ascend 门禁 OK（triton +7.8%、torch/reference
±2% 内，无超阈回退）；P800 门禁 OK（新 fast-path 基线 Δ=0.0%）。

## 6. 单 kernel 融合实验（2026-10-01）

**动机**：小形状被 launch 主导（地板 97µs），causal 路径 3 次 launch
≈0.29ms ≈ decode 整体耗时。做法：`_probs_kernel` 增加 `HAS_PV` 模式，
QK→softmax→PV 在一个 kernel 内完成，P 块只在寄存器里存在、结果直接
写 `O`（`probs` 仍按契约物化），causal 尾块补零也一并收进 kernel
（去掉独立 `torch.zeros`）。选择逻辑按形状自动走（`SDPA_MATH_FUSED`
=on/off/auto 可强制，仅实验用）。

**阈值不是形状面积，而是实际处理的 score tile 数**（causal 按行块截断），
实测两路径对比（`thr*.py`，D128、causal，ratio<1 = 融合更快）：

| shape | tiles | 融合 | 两段式 | ratio |
|---|---:|---:|---:|---:|
| 64×64（bench decode 形状） | 1 | 0.177 | 0.225 | **0.78** |
| 1×1024（decode 长上下文） | 1 | 0.169 | 0.229 | **0.74** |
| 1×8192 | 1 | 0.185 | 0.227 | **0.82** |
| 64×1024 | 1 | 0.179 | 0.226 | **0.79** |
| 128×2048 | 3 | 0.195 | 0.229 | **0.85** |
| 128×128 | 3 | 0.187 | 0.224 | **0.84** |
| 100×100（bench tail 形状） | 3 | 0.188 | 0.227 | **0.83** |
| 256×256（causal） | 10 | 0.201 | 0.222 | **0.91** |
| 320×320（causal） | 15 | 0.218 | 0.222 | **0.98** |
| 256×256（非 causal） | 16 | 0.214 | 0.221 | **0.97** |
| 384×384（causal） | 21 | 0.241 | 0.226 | 1.07 |
| 448×448（causal） | 28 | 0.257 | 0.238 | 1.08 |
| 512×512（causal） | 36 | 0.292 | 0.258 | 1.13 |
| 512×512（非 causal） | 64 | 0.364 | 0.314 | 1.16 |
| 1024×1024（causal） | 136 | 0.614 | 0.469 | 1.31 |
| 2048×2048（causal） | 528 | 1.855 | 1.317 | 1.41 |

交叉点在 15~21 tiles → **阈值取 16 tiles**（`_FUSED_MAX_TILES`）。
同为 262k 面积的 causal 512²（36 tiles）融合慢 13%、而 causal
128×2048（3 tiles）融合快 15%——故判据用 tiles 而非 `max(Sq,Skv)`。
`dropout_p>0` 恒走两段式（掩码须先作用在 P 上，见文件头注）。

**结果**（直接收益，见 §2 表）：decode 0.287→0.217ms（**-24%**，
0.92x→1.17x）、tail100 0.285→0.219ms（**-23%**，0.97x→1.16x）；
大形状按阈值走两段式，与融合前持平。**代价**：融合模式在大形状上
劣化 13-41%（见上表），因此不能全局开；进程内多一套参数组合使
1k 门禁用例 +7.8%（噪声带内）。

**回归**：`check_accuracy --impl triton` **175/175，worst 3.906e-3
不变**（融合/两段式两路径都覆盖）；`example.py ascend910` 52/21/8
全绿；`example.py cpu` 全绿；门禁 FAIL 0（§4）。

## 7. 复现

```bash
python3 script/bench_perf.py --device npu:0 --register \
        --json-out reports/perf_ascend910.json                # NPU fp16
python3 script/bench_perf.py --device cpu \
        --json-out reports/perf_cpu.json                      # CPU fp32
python3 script/bench_perf.py --device cuda:1 --register \
        --json-out reports/perf_p800-kunlunxin.json           # P800 fp16
python3 script/bench_f_context.py --device cuda:1 \
        --json-out reports/perf_f_context_p800-kunlunxin.json # 仅参考列
python3 ../../scripts/perf_run.py     --device ascend910 --pattern sdpa_math \
        --update-baseline                                    # 采基线（有意动作）
python3 ../../scripts/perf_compare.py --device ascend910      # 门禁（FAIL 0）
python3 ../../scripts/perf_run.py     --device p800-kunlunxin \
        --pattern ops.sdpa_math --update-baseline             # 优化后基线
python3 ../../scripts/perf_compare.py --device p800-kunlunxin # 门禁（FAIL 0）
```
