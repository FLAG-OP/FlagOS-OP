# 算子测试报告: clone

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `clone` / A1 / Triton 级 |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过**（应用层为等价 torch API 验证） |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层 | ✅ | `test/kernel_level.py cambricon` | PASS |
| 框架层 | ✅ | `test/op_level.py cambricon` | PASS |
| 注册契约 | ✅ | PR #25 / `register.py` | `counter` + `platform`；canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`；包态导入优先 |
| 包态执行 | ✅ | `test/op_level.py` 子进程 | `ops.clone.register → ops.clone.kernel`，输出存储独立 |
| 应用层 | ✅（等价） | `test/framework_level.py cambricon` | PASS |
| 黄金回归 | ✅ | `script/` 两脚本 | 48/48 |
| 性能门禁 | ✅ | `scripts/perf_run.py --pattern ops.clone` | 基线可建 |

## 2. 环境

`check_env --device cambricon` → OK（版本全一致）。

## 3. 精度结果

| dtype | 容差 | max err | 判定 |
|---|---|---|---|
| float32 | 1e-5 | 0.0 | ✅ |
| float16 | 1e-2 | 0.0 | ✅ |
| bfloat16 | 1e-2 | 0.0 | ✅ |

- 黄金 48 组（含非整倍数 + zeros/large/boundary）全通过。
- 存储独立: 修改 clone 不影响 src。
- 哨兵: 确定性 ✅ 敏感 ✅。
- 视图: 转置/permute/步长/expand/channels_last，err 0 且 stride 与原生一致。

## 4. 性能结果

8192² fp32: 自研 0.242ms / torch 级 0.238ms / 原生 0.238ms。
FlagGems 无 clone 算子。自研与原生持平。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | vLLM/transformers 未安装 | 应用层非真实推理 | 已注明 |
| 2 | 性能无提升空间（内存带宽受限） | — | 可接受 |

## 6. 结论

Must 项（适用层验证通过（C1 等价 torch API）、黄金、位级一致、存储独立）**通过**。
