# 算子总体报告: dropout

| 项 | 值 |
|---|---|
| 算子名称 | `dropout` |
| 语义 | [reference.py](reference.py)（原生 F.dropout；随机分支按结构/统计判定） |
| 硬件平台 | **2 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`）· p800-kunlunxin（P800 / XMLIR；A1 canonical `AutogradCUDA`，兼容 `CUDA`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ `p800-kunlunxin`（P800）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 / 2026-10-22 P800 交付 / 2026-10-26 复验 |

## p800-kunlunxin 接入（2026-10-22 交付 / 2026-10-26 复验）

| 项 | 结论 |
|---|---|
| 设备呈现 | XMLIR 将 XPU 呈现为 CUDA；`device.type == "cuda"` |
| A1 拦截 key | `AutogradCUDA`（兼容 `CUDA`；profile `dispatch_key=CUDA`） |
| 设备探测 | `import torch_xmlir` + `torch.cuda.is_available()` |
| 实现路径 | Triton（P800 随机分支委托原生） |
| 三层验证 | `python3 example.py p800-kunlunxin` 全绿；op 层拦截 5 次 |
| 黄金 | 189/189，0 fail |
| 性能门禁 | 空闲 P800 两轮复测：`perf_compare --device p800-kunlunxin` = OK（FAIL 0 / WARN 0 / NEW 0） |

| 性能用例 | 延迟 (ms) |
|---|---:|
| reference | 0.385 |
| torch | 0.385 |
| triton | 0.385 |

**P800 绑定结论**：自研 Triton 0.385 ms = 原生 0.385 ms（1.00x）；P800 随机分支委托原生，channels_last 已还原。

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | [kernel/triton_level.py](kernel/triton_level.py) | ✅ Philox |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | ⬜ 置空 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 | 详细 |
|---|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ | — |
| kernel 层 | `test/kernel_level.py` | ✅ | [reports/accuracy.md](reports/accuracy.md) |
| 框架层 | `test/op_level.py` | ✅ | — |
| 应用层 | `test/framework_level.py` | ✅（等价 torch API） | — |
| 黄金 | `script/gen_golden.py` + `check_accuracy.py` | ✅ 189/189 | [goldendata/](goldendata/README.md) |
| 性能 | perf 门禁 | ✅ | [reports/performance.md](reports/performance.md) |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `dropout` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops.dropout import register_a1`。
- PR #25 聚合复验覆盖 MLU 基础算子契约与代表性真实 vLLM A1 注入；本算子目录的应用层仍是等价 torch API，不能据此宣称本算子已由真实 vLLM 单独消费。

## 关键数字

| 指标 | 自研 Triton | FlagGems | 原生 |
|---|---|---|---|
| 确定性分支精度 | 位级一致 | — | 0 |
| 随机 drop 偏差 | 0.0016 | — | — |
| 8192² fp16 p=0.5 延迟 | 2.785 ms | 2.960 ms | **0.563 ms** |
| seed 可控 / 哨兵 | ✅ / ✅ | — | — |

## 结论与遗留

适用层验证通过（C1 等价 torch API）、黄金 189/189；确定性分支位级一致、随机分支统计合规、
`torch.manual_seed` 可控。性能受 Philox RNG 限制，比原生慢 ~5x，
但与 FlagGems 同级（略快）。遗留: 真实 vLLM 应用层、RNG 向量化调优。

## 交付物清单

| 交付物 | 位置 |
|---|---|
| 开发报告 | [reports/development.md](reports/development.md) |
| 测试报告 | [reports/test-report.md](reports/test-report.md) |
| 精度分册 | [reports/accuracy.md](reports/accuracy.md) |
| 性能分册 | [reports/performance.md](reports/performance.md) |
