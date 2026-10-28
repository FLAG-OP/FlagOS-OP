# thead-ppu 移植报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> 任务: T3 平头哥 thead-ppu（PPU-ZW810E）移植。任务书 §2 a→i。
> 日期: 2026-10-24 · 提交: wt-2026-10-24-fix
> 环境: `flagos-ppu-wt-dev` · torch 2.10.0 · triton 3.6.0 · CUDA 兼容层
> 用卡: cuda:6,7（避 4/5/11）· 16 卡均为 PPU-ZW810E

## 1. 分叉判定（§2a）

| 算子 | 结论 | 理由 |
|---|---|---|
| T1 `ops/sdpa` | 结构级 | 新文件 `kernel/thead_fast_level.py`、函数 `sdpa_math_thead_fast`。两段式 probs+PV 在 PPU flash 上无对应路径，需独立路由 |
| T2 `ops/embedding` | 参数级 | 已有 thead catalog 条目，直接沿用 T1-sdpa 的 kernel 本体 |
| T3 `ops/sdpa_math` | **结构级** | 同 T1；须物化第二输出 P（不整体换 flash/efficient）；同构 p800 vendor O/LSE + exact-P，差异在路由与 LSE 布局，见 PLATFORM.md §3a |

T1 的绑定结论全部迁入 sdpa_math 的 thead 章节（PLATFORM.md §3a），
本报告仅记录 sdpa_math 专属证据。

## 2. 环境与首测（§2b）

- 设备: 16 卡均 PPU-ZW810E（`ppu-smi` 均显示该型号）；用卡 6,7（2MiB 各）。
- 栈: torch 2.10.0（CUDA 兼容层）· triton 3.6.0（PPUDriver）· 无 flag_gems。
- `PYTHONPATH=/hosthome/FlagOS-OP`；测试设备 `SDPA_MATH_TEST_DEVICE=cuda:6`。

首测探针 `probes/thead_first_probe.py`（每 key 独立进程）：

| 项 | 结果 |
|---|---|
| exact-P | fp16/bf16 maskless（flash）、masked（efficient）、GQA、bool/float mask 全达标 |
| LSE shape | flash: `(B,H,Sq)` 精确；efficient: 按 32 倍数 pad（16→32, 100→128, 197→224） |
| flash 非方阵 causal 缺陷 | flash O err≈2.0 vs efficient err≈0 — 非方阵 causal 强制走 efficient |
| flash dtype | 仅 fp16/bf16；fp32 直调报错 |
| 注册 key | AutogradCUDA 6/7（唯一零命中 `inference_mode`，同 known-issues #19）；纯 CUDA 7/7 但有 autograd 警告 → 选 AutogradCUDA |

证据归档: `probes/out/thead_first_{exact_p,key_AutogradCUDA,key_CUDA}.json`。

## 3. 实现决策（§2c）

| 决策 | 结论 |
|---|---|
| 路由 | maskless 且（无 causal 或 Sq==Skv）→ flash；带 mask 或非方阵 causal → efficient |
| LSE 处理 | flash 精确；efficient 先 `lse[..., :Sq]` 切片再 materialize P |
| GQA | flash/efficient schema 均无 `enable_gqa` → `repeat_interleave` 扩 KV |
| bool_b1 头广播 mask | efficient 要求 bias 头维与 Hq 对齐 → `(B,1,Sq,Skv)` 先 `expand` 再进 kernel |
| 16B stride | `_aligned_bias` 列 pad（p800 同款） |
| 全遮蔽行 | vendor LSE=0、out=0；naive exp(score−LSE) 出垃圾 P — mask/causal 零填充后 P=0 |
| fp32 | 全量 torch_level ATen 组合。mild 委托 err 3.6e-7 达标，但 extreme scale=8 委托 abs≈2e-5 超 1e-5 判卷线（T1-sdpa/p800 同判） |
| bf16 boundary | V≈dtype_max/2 时厂商 kernel O 中间量溢出（原生同溢出）；fp32 torch 路径可保持有限。tail 判定: 有限子集 rel + inf parity |

## 4. 验证（§2d-h）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `test/kernel_level.py thead-ppu` | **52/52** · max err 3.9e-3 bf16 |
| tail 直测 | `test/kernel_level_tail_thead_ppu.py` | **47/47** · 3072/5000/5120/5000×5001 tails × fp32/bf16/fp16 × seeds 11/22/33 + zeros/large/boundary + 4096 floatmask |
| 框架层 | `test/op_level.py thead-ppu` | **21/21** · intercepts=3822 · gradcheck 6 cases |
| 应用层 | `test/framework_level.py thead-ppu` | **8/8** · intercepts=21 · logits/grad L∞=0 |
| 黄金回归 | `script/check_accuracy.py --impl thead --device cuda:6` | **175/175** · worst 1.050e-2 abs（bf16 容差 2e-2） |
| 一键 | `python3 example.py thead-ppu` | 三层全绿 |
| perf 门禁 | `scripts/perf_run.py --device thead-ppu --pattern ops.sdpa_math --update-baseline` | **FAIL 0 · WARN 0** · 8 条全 OK |
| 跨层一致性 | `run.py --consistency --device thead-ppu` | **缺证** ModuleNotFoundError: flag_gems（结构性，同 T1/T2） |

### 性能（§2h）

`perf/baselines/thead-ppu.json`（基线新增，T1/T2 已动两轮，本 commit 说明原因）:

| case | latency_ms | TFLOPS |
|---|---|---|
| `ops.sdpa_math.thead-ppu` | 0.328 | 13.10 |
| `ops.sdpa_math.native` | 0.782 | 5.49 |
| `ops.sdpa_math.torch` | 0.763 | 5.63 |
| `ops.sdpa_math.reference` | 1.147 | 3.75 |

**direct 2.38x native composite**（同为返回 out+P 的公平口径）。
bench 六形状 0.97-1.06x（vs 注册后原生——A1 同路径）。

## 5. 交付清单（§2i）

| 文件 | 变更 |
|---|---|
| `kernel/thead_fast_level.py` | 新增 |
| `_profile.py` | thead-ppu profile entry |
| `register.py` | `_IMPLS["thead"]` + import |
| `__init__.py` | `sdpa_math_thead` export |
| `test/kernel_level.py` | `_impl_for` thead branch |
| `test/kernel_level_tail_thead_ppu.py` | 新增 |
| `test/op_level.py` | `impl_fn` dict thead entry |
| `script/check_accuracy.py` | IMPLS thead/thead-ppu |
| `example.py` | thead perf_cases branch |
| `probes/thead_first_probe.py` | 新增 |
| `probes/out/thead_first_*.json` | 证据归档 |
| `PLATFORM.md` | 新增 §3a thead section |
| `MERGE.md` | 绑定速查更新 |
| `README.md` | 平台列 + tree 更新 |
| `REPORT.md` | 实现矩阵 + 验证矩阵 + 关键数字 |
| `reports/thead-ppu.md` | 本报告 |
| `reports/development.md` | thead 环境段 |
| `reports/test-report.md` | thead 行 |
| `reports/performance.md` | thead 段 |
| `ops/catalog.yaml` | thead profiles entry |

## 6. 遗留

- `run.py --consistency` 锚定 gelu_and_mul（A2）硬依赖 flag_gems — dev 容器无，结构性缺证（T1/T2 同根因）
- FlagOS E2E `script/e2e_flagos.py` 未跑 — 依赖 flag_gems 栈，thead-ppu 无
- T2 `ops/embedding` thead 用例已在 catalog，直接沿用 T1-sdpa kernel
- 跨层一致性锚定 `ops.sdpa_math` 的 thead 端到端已由 op/framework 层拦截覆盖

## 7. 复现命令

```bash
cd /hosthome/FlagOS-OP
export PYTHONPATH=/hosthome/FlagOS-OP
export SDPA_MATH_TEST_DEVICE=cuda:6

# kernel
python3 ops/sdpa_math/test/kernel_level.py thead-ppu
python3 ops/sdpa_math/test/kernel_level_tail_thead_ppu.py

# op / framework
python3 ops/sdpa_math/test/op_level.py thead-ppu
python3 ops/sdpa_math/test/framework_level.py thead-ppu

# 黄金
python3 ops/sdpa_math/script/check_accuracy.py --impl thead --device cuda:6

# 一键
python3 ops/sdpa_math/example.py thead-ppu

# perf
python3 ops/sdpa_math/script/bench_perf.py --device cuda:6 --register \
  --json-out ops/sdpa_math/reports/perf_thead-ppu.json
python3 scripts/perf_run.py --device thead-ppu --pattern ops.sdpa_math
python3 scripts/perf_compare.py --device thead-ppu
```
## 8. C2 真实 vLLM（thead-ppu-c2）

2026-10-28 在真实引擎基座补齐 C2 证据：默认 flagos/FA3 路径**不**消费
math composite；经显式 MATH 路径（sitecustomize 改
`ALL_ATTENTION_FUNCTIONS["vllm"]`）在 EngineCore 内 A1 命中 **20** 次，
token 与同路径基线一致。详见 [thead-ppu-c2.md](thead-ppu-c2.md) 与
[e2e_vllm_c2_thead-ppu-c2.json](e2e_vllm_c2_thead-ppu-c2.json)。
