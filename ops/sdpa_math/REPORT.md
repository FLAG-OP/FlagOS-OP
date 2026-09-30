# 算子总体报告: sdpa_math（`aten::_scaled_dot_product_attention_math`）

> 定位: 一页看全一个算子——实现矩阵、验证矩阵、关键数字。
> 分册在 [reports/](reports/)。

| 项 | 值 |
|---|---|
| 算子名称 | `aten::_scaled_dot_product_attention_math`（SDPA **math 后端**，双输出 `(out, attn_probs)`） |
| 语义 | [reference.py](reference.py)（GQA / causal / bool·float mask / dropout 双规则 / fp32 内部） |
| 目标设备 / 路线 | `ascend910` + `cpu` / **A1** ☑（aten 拦截，成对注册 `Autograd*` + 纯设备键） |
| 日期 / 状态 | 2026-09-30 / 定稿 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 | 备注 |
|---|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ☑ | 独立 ATen 组合（`nan_to_num` softmax 守卫），CPU 交付实现 + 第二判卷人 |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ☑ | 两段式 `_probs_kernel`(pass1/2 共用 `_score_block`) + `_pv_kernel`；dropout 用 ATen 收口 |
| 硬件级 | `kernel/hardware_level/` | — | 未做：CANN/厂商绑定不在本交付范围（torch 级已覆盖 CPU） |

`reference.py` 为第三份独立实现（语义判卷标准），三者逐条对齐并同过黄金。

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细报告 |
|---|---|---|---|
| 一键三层 | `python3 example.py ascend910` / `cpu` | ☑ 全绿（52+21+8 项） | — |
| kernel 层 | `test/kernel_level.py` | ☑ 52 组 ×2 profile，哨兵/dropout/错误路径全过 | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ☑ 21 项：四模式拦截、注册=直调（逐位）、6 组 gradcheck、F.sdpa(MATH) 拦截 | [reports/test-report.md](reports/test-report.md) |
| 应用层 | `test/framework_level.py` | ☑ 8 项：mini-decoder + 概率图消费者双跑，top1=1.0、贪心序列 1.00 | [reports/test-report.md](reports/test-report.md) |
| 黄金回归 | `script/gen_golden.py` + `check_accuracy.py` | ☑ **175/175**（reference/torch/triton 三实现）；原生对照 **131/131**（44 bool 跳过） | [reports/accuracy.md](reports/accuracy.md) |
| 性能 | `script/bench_perf.py`（含 `--register` A1 路径） | ☑ NPU prefill/GQA 1.20-2.05x；CPU 1.05-1.40x | [reports/performance.md](reports/performance.md) |
| 语义证据 | `probes/native_semantics.py` | ☑ 6 节全过（两 profile） | [reports/development.md](reports/development.md) |
| perf 门禁 | `common/perf_registry.py` → `scripts/perf_run.py --pattern sdpa_math` | ☑ 已登记 `ops.sdpa_math` | [reports/performance.md](reports/performance.md) |

## 关键数字

| 指标 | 值 | 对照 |
|---|---|---|
| 黄金最差 abs | 3.91e-3（triton/bf16）· 1.95e-3（torch/native/bf16） | 容差 bf16=2e-2；fp32 三实现 ≤1.2e-7 |
| 原生一致性 | 131/131 PASS，worst 9.77e-4（NPU） | `--impl native`，bool 用例 44 组按规则跳过 |
| 梯度 | 6 组 fp64 gradcheck（base/causal/float-mask±grad/gqa/explicit-dropout/dP 消费者） | 数值差分 |
| 延迟（NPU fp16 vs 原生 math） | 2k D128 **1.316ms / 2.692ms = 2.05x**；1k D128 1.32x；GQA 1.74x；decode 0.92x | 同为"返回 out+概率图"的公平口径 |
| A1 包装开销 | NPU 1k D64 直调 0.528ms → A1 0.618ms（≈0.09ms autograd.Function 包装） | 仍 1.03x 于原生 |
| 哨兵 | 确定性 ☑ 敏感 ☑（dropout 随机路径同种子可复现 ☑） | |

## 结论与遗留

**可交付**: 三层全绿、黄金与原生对照双过、A1 拦截在 grad/no_grad/
inference_mode/无 requires_grad 四种模式全部命中，梯度（含 `dprobs`
通路）通过 gradcheck 与参考反传双验证。

遗留（有意为之，均有注释与证据）:
1. `bool attn_mask` **直调**语义按 `-inf` 遮蔽实现（对齐 `F.sdpa`），
   与 native 直调的 0/1 加性怪癖有意分歧——44 组黄金在 `--impl native`
   下跳过；走 `F.sdpa` 的消费方行为完全一致。
2. decode/tail 小形状相对原生略慢（0.92-0.97x）：本算子必须物化
   `(B,Hq,Sq,Skv)` 概率图，小形状下 kernel 启动占比高。
3. `dropout_p>0` 且走 A1 路径时，`out=(P)@V` 用 ATen matmul 而非
   fused PV（dropout 不是本算子的性能路径，`register.py` 已注明）。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告（7 章） | [reports/development.md](reports/development.md) |
| 测试报告（范围矩阵） | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
| native 语义证据 | [probes/native_semantics.py](probes/native_semantics.py) |
| 性能原始数据 | [reports/perf_ascend910.json](reports/perf_ascend910.json) · [reports/perf_cpu.json](reports/perf_cpu.json) |
| 仓库已知问题登记（六类语义/注册分化） | [docs/known-issues.md #19](../../docs/known-issues.md) |
