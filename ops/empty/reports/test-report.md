# 算子测试报告: empty

> **环境口径**：本报告记录 2026-09-18 的历史交付快照（Python 3.10.20、torch 2.7.1、torch_mlu 1.29.2、Triton 3.2.0）。当前复验环境与平台支持范围以 [软硬件环境版本汇总](../../../docs/environments.md)、[算子目录](../../../ops/README.md) 和当前 `REPORT.md` 为准；报告中的 `PrivateUse1` 结果属于兼容性记录，当前 canonical key 为 `AutogradPrivateUse1`。


| 项 | 值 |
|---|---|
| 算子 / 路线 / 级别 | `empty` / A1 / torch 级（分配/元数据） |
| 目标设备 / 日期 | `cambricon`（MLU590-M9）/ 2026-09-18；2026-10-12 契约复验 |
| 结论 | **通过** |

## 1. 测试范围

| 层级 | 覆盖 | 入口 | 结果 |
|---|---|---|---|
| kernel 层（元数据） | ✅ | `test/kernel_level.py` | PASS（15 项） |
| 框架层（注册/拦截） | N/A | `test/op_level.py` | 按设计不注册 |
| 注册契约 | ✅ | PR #25 / `register.py` | 签名对齐；仍按设计拒绝 A1 注册 |
| 应用层 | ✅ | `test/framework_level.py` | PASS |
| 黄金（元数据） | ✅ | `script/` | 48/48 |
| 数值精度/哨兵 | ⬜ 不适用 | — | 未初始化内存 |

## 2. 环境

`check_env --device cambricon` → OK。

## 3. 元数据结果

| 维度 | 覆盖 | 结果 |
|---|---|---|
| shape/dtype/device | 多 shape × dtype | ✅ |
| stride | 连续 / 非连续 / 例外 stride | ✅ |
| 零元素 | `(0,5)` | ✅ |

## 4. 性能结果

8192² fp32 分配 0.0107ms（分配器受限，非门禁项）。

## 5. 问题与风险

| # | 描述 | 影响 | 状态 |
|---|---|---|---|
| 1 | 无数值语义 | 无精度/哨兵 | 语义使然 |
| 2 | vLLM 未安装 | 应用层等价验证 | 已注明 |

## 6. 结论

适用层 Must 项（元数据一致）**通过**；注册/拦截层按设计为 N/A。

## 7. P800（p800-kunlunxin）复验（2026-10-26）

| 层级 | 入口 | 结果 |
|---|---|---|
| kernel 层 | `python3 test/kernel_level.py p800-kunlunxin` | PASS |
| 框架层 | `python3 test/op_level.py p800-kunlunxin` | PASS；按设计不注册 |
| 应用层 | `python3 test/framework_level.py p800-kunlunxin` | PASS（等价 torch API 子进程） |
| 黄金 | `python3 script/check_accuracy.py --impl torch --device cuda:1` | 48/48，0 fail |
| 性能门禁 | `python3 scripts/perf_compare.py --device p800-kunlunxin` | OK |

结论：P800 复验 **通过**。不注册以避免与 empty_strided 递归；P800 仅 kernel/应用等价验证。
