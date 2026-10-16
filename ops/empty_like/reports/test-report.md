# 算子测试报告: empty_like

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `empty_like` / A1 / torch 级（分配/元数据） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（元数据） | ✅ | `test/kernel_level.py` | PASS（50 项） |
| 框架层（注册/拦截） | ✅ | `test/op_level.py` | PASS（3 次） |
| 注册契约 | ✅ | PR #25 / `register.py` | `counter` + `platform`；canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`；包态导入优先 |
| 应用层 | ✅（等价） | `test/framework_level.py` | PASS |
| 黄金（元数据） | ✅ | `script/` | 168/168 |
| 数值精度/哨兵 | ⬜ 不适用 | — | 未初始化内存 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 元数据结果

| 维度 | 覆盖 | 结果 |
|---|---|---|
| shape/dtype/device | 继承 self + dtype/device 覆盖 | ✅ |
| stride | preserve_format / contiguous_format / channels_last | ✅ |
| 存储独立 | data_ptr 不同 | ✅ |
| 零元素 | `(0,5)` | ✅ |

## 4. 性能结果

8192² fp32 分配 0.0133ms（分配器受限，非门禁项）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无数值语义 | 无精度/哨兵 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

Must 项（适用层验证通过（C1 等价 torch API）、元数据一致、注册拦截、存储独立）**通过**。
