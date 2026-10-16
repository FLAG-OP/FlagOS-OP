# 算子测试报告: contiguous

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `contiguous` / A1 / Triton 级 |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层 | ✅ | `test/op_level.py cambricon` | PASS |
| 注册契约 | ✅ | PR #25 / `register.py` | `counter` + `platform`；canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`；包态导入优先 |
| 应用层 | ✅（等价） | `test/framework_level.py cambricon` | PASS |
| 黄金回归 | ✅ | `script/` 两脚本 | 114/114 |
| 性能门禁 | ✅ | `scripts/perf_run.py --pattern ops.contiguous` | 可建基线 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 精度结果

| 输入布局 | dtype | max err | 连续性 | 别名 | 判定 |
|---|---|---|---|---|---|
| 连续 | fp32/fp16/bf16 | 0 | ✅ | ✅ 返回 self | ✅ |
| transpose/permute/slice/expand | fp32/fp16/bf16 | 0 | ✅ | 新存储 | ✅ |

黄金 114 组（contig + view 两种 layout，含非整倍数/特殊用例）全通过。
哨兵: 确定性 ✅ 敏感 ✅。

## 4. 性能结果

8192² fp32 转置→连续: 自研 0.305ms / 原生 0.254ms（慢 ~20%）。
修复前自研为 24.7ms（逐元素 gather），加 2D swiz 分块后提升 ~80x。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | rank≥3 未分块 | 较慢 | 待优化 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

Must 项（适用层验证通过（C1 等价 torch API）、黄金、别名/连续性正确）**通过**。
