# 算子测试报告: type_as

| 项 | 值 |
|---|---|
| 算子名称 / 路线 / 级别 | `type_as` / A1 / Triton 级（+ torch 级对照） |
| 目标设备 / 测试日期 | `cambricon`（MLU590-M9）/ 2026-09-18 |
| 结论 | **通过**（应用层为等价 torch API 验证，非真实 vLLM） |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（精度/哨兵/性能） | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层（注册/拦截） | ✅ | `test/op_level.py cambricon` | PASS |
| 应用层 | ✅（等价） | `test/framework_level.py cambricon` | PASS（vLLM 未安装） |
| 黄金回归 | ✅ | `script/gen_golden.py` + `check_accuracy.py` | 111/111 |
| 性能回归门禁 | ✅ | `scripts/perf_compare.py --device cambricon` | 基线已建 |
| 跨层一致性 | — | 仓库 `--consistency`（锚定 gelu_and_mul） | 不适用本算子 |

一键: `python3 example.py cambricon`

## 2. 环境

`python3 scripts/check_env.py --device cambricon` → **结论: OK**（torch/
torch_mlu/triton/flag_gems/numpy/python 全部锁定版本一致）。

## 3. 精度结果

| 输出 dtype | 容差 | max err | 判定 | 对照 FlagGems `to_copy` | 对照原生 |
|---|---|---|---|---|---|
| float32 | 1e-5 | 0.0 | ✅ | 0.0 | 0.0 |
| float16 | 1e-2 | 0.0 | ✅ | 0.0 | 0.0 |
| bfloat16 | 1e-2 | 0.0 | ✅ | 0.0 | 0.0 |

- 黄金: 111 组（含非 BLOCK 整倍数 14336/5000/5120/4097；zeros/large/boundary 特殊用例）全部通过。
- 哨兵: 确定性 ✅ · 输入敏感 ✅（排除静默 no-op）。
- 非 dense 视图（转置/permute/步长/expand/channels_last）: err 0 且 stride 与原生一致。

## 4. 性能结果

8192×8192 fp16→fp32（短采样 20+100 + `torch.mlu.synchronize`）:

| 实现 | 延迟 | 带宽 | 相对自研 |
|---|---|---|---|
| 自研 Triton | 0.190 ms | 2122 GB/s | 1.00x |
| PyTorch 原生 `.to` | 0.175 ms | 2300 GB/s | 1.09x |
| FlagGems `ops.to_copy` | 0.317 ms | 1270 GB/s | 0.60x |

微算子分发开销: A1 aten 路径 ≈3µs（[集成指南](../../../docs/integration.md)），
相对 0.19ms 的 cast 可忽略。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 非 dense 窄→宽 gather+widening store MLU codegen 崩溃 | 走 `out.copy_` 兜底 | 已知，待上游修复 |
| 2 | 自研 Triton 慢于原生 ~9% | 性能非最优 | 可优化 |
| 3 | vLLM/transformers 未安装 | 应用层非真实推理 | 环境限制，已注明 |

## 6. 结论

对 Must 分级（三层全绿 + 黄金 + 精度/O 回退）**通过**；黄金 111/111，
精度位级一致，性能满足微算子需求。遗留：真实 vLLM 应用层、连续 cast
性能优化、非 dense 窄→宽两阶段 kernel（见
[reports/development.md](development.md) §7）。
